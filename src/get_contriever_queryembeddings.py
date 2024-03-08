#import os
import torch
import torchvision
import tarfile
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import copy
import math
from torch.nn import init
import transformers
import argparse
import wandb
from torch import Tensor, nn

from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling,AutoTokenizer
from transformers import AutoModelForCausalLM
import datasets
from pathlib import Path
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist

import os 

import argparse

datasets.config.DOWNLOADED_DATASETS_PATH = Path('/var/local/pbansal/huggingface/bookcorpus')
datasets.config.HF_DATASETS_CACHE = Path('/var/local/pbansal/huggingface_cache')

parser = argparse.ArgumentParser()
parser.add_argument('--save', action='store_true')
parser.add_argument('--ddp', action='store_true')
parser.add_argument('--save_ckpt_name', type=str,default='')
parser.add_argument('--load_ckpt_name', type=str,default='')
parser.add_argument('--batch_size', type=int,default=32)
parser.add_argument('--loss_batch_size', type=int,default=32)
parser.add_argument('--test_batch_size', type=int,default=64)
parser.add_argument('--num_warmup_steps', type=int,default=10000)
parser.add_argument('--evaluation_every_kiterations', type=int,default=2000)
parser.add_argument('--num_iterations', type=int,default=-1)
parser.add_argument('--num_epochs', type=int,default=1)
parser.add_argument('--max_seq_len', type=int,default=64)
parser.add_argument('--exit_at', type=int,default=1)
parser.add_argument('--seed', type=int,default=0)


args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# os.environ['MASTER_PORT'] = 12345

# tokenized_dataset = datasets.load_from_disk('/var/local/pbansal/huggingface/bookcorpus_tokenized_bert')['train']
tokenized_dataset = datasets.load_from_disk('/var/local/pbansal/huggingface/openwebtext_tokenized_bert')['train']


length_dataset = len(tokenized_dataset)
# assert length_dataset == 74004228
tokenized_dataset = tokenized_dataset.train_test_split(0.0001,seed=0)
train_size = len(tokenized_dataset['train'])
test_size = len(tokenized_dataset['test'])

test_size = args.test_batch_size*(int(test_size/args.test_batch_size))
# assert train_size == 73996827


if args.ddp:
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank
else : 
    device = 'cuda:0'
    master_process = True
    seed_offset = 0


data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

model = lm.BERTModel().to(device)
if (args.ddp):
    model = DDP(model, device_ids=[ddp_local_rank],find_unused_parameters=True)
if (args.load_ckpt_name != ''):
    model.load_state_dict(torch.load(args.load_ckpt_name,map_location=device))

model.module.wte.weight = torch.nn.Parameter(torch.load('/var/local/pbansal/gsgd/bert_word_embeddings.pt').weight.cuda()).requires_grad_(False)
model.module.wpe.weight = torch.nn.Parameter(torch.load('/var/local/pbansal/gsgd/bert_position_embeddings.pt').weight.cuda()).requires_grad_(False)

number_of_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)


max_lr = 1e-4
weight_decay = 1e-2
epoch_ = 1

batch_size = args.batch_size
exit_at = args.exit_at
loss_batch_size = args.loss_batch_size
num_warmup_steps = args.num_warmup_steps
skip_exit = (loss_batch_size >=batch_size)

if (args.num_iterations == -1):
    num_epochs = args.num_epochs
    num_iterations = int(train_size/batch_size)*num_epochs
else : 
    num_iterations = args.num_iterations
    num_epochs = math.ceil(num_iterations/int(train_size/batch_size))

if (master_process):
    print ("Running for %d Epochs, %d Iterations"%(num_epochs,num_iterations))
    print ("Number of Trainable Parameters %d"%number_of_trainable_params)


if (master_process):
    wandb.login(key="e45d2f6c4df62f742cc5974e9865de8bfeaacc00")
    wandb.init(project='gsgd_mlm', entity="pbansal", dir = '/var/local/pbansal/dumps/peft')

    if skip_exit:
        str_ = ''
    else :
        # str_ = '_loss_exitat%d'%args.exit_at if args.loss_based else '_entropy_exitat%d'%args.exit_at
        str_ = '_loss_exitat%d'%args.exit_at

    wandb.run.name = 'owt_%d_%d_%dK%s'%(batch_size,loss_batch_size,int(num_iterations/1000),str_)
    if (args.save):
        wandb.run.name = 'owt_pretraining_run_%d_%dK'%(batch_size,int(num_iterations/1000))


optimizer = torch.optim.AdamW([x for x in model.parameters() if x.requires_grad], max_lr,weight_decay=weight_decay)
if (args.save):
    sched = transformers.get_constant_schedule_with_warmup(optimizer,num_warmup_steps=num_warmup_steps)
    # sched = transformers.get_linear_schedule_with_warmup(optimizer,num_warmup_steps=num_warmup_steps,num_training_steps=num_iterations)
else : 
    sched = transformers.get_linear_schedule_with_warmup(optimizer,num_warmup_steps=num_warmup_steps,num_training_steps=num_iterations)

loss_fct = nn.CrossEntropyLoss(reduction='none')
softmax = nn.Softmax(dim=-1)

np.random.seed(args.seed+seed_offset)

train_sample_batch_order = np.random.permutation(train_size)
batch_order_index = 0

for iter_ in range(1,num_iterations+1):
    to_log = {'train/iteration':iter_, 'train/epoch':epoch_}

    batch_indices = [train_sample_batch_order[int(i)] for i in np.arange(batch_size)+batch_order_index]
    batch_data = data_collator([tokenized_dataset['train'][int(i)]['input_ids'][:args.max_seq_len] for i in batch_indices])
    batch_data['mask_ids'] = torch.bernoulli(torch.zeros(batch_data['input_ids'].shape)+0.15)
    batch_data['input_ids'][batch_data['mask_ids'] == 1] = 103


    batch_order_index += batch_size
    
    if (batch_order_index+batch_size>train_size):
        train_sample_batch_order = np.random.permutation(train_size)
        batch_order_index = 0
        epoch_ += 1

    if (skip_exit):
        selected_train_indices = np.arange(batch_size)
    else : 
        with torch.no_grad():
            lm_logits = model(batch_data['input_ids'].pin_memory().to(device, non_blocking=True),hidden_states = None,exit_at = exit_at)
            labels = batch_data['labels'].pin_memory().to(device, non_blocking=True)

            loss = loss_fct(lm_logits.transpose(1,2), labels)
            loss *= batch_data['mask_ids'].pin_memory().to(device, non_blocking=True)

            loss = loss.sum(axis=1)/(loss>0).sum(axis=1)
            # loss = loss.sum(axis=1)

            selected_train_indices = torch.argsort(loss)[-loss_batch_size:].cpu()

    lm_logits = model(batch_data['input_ids'][selected_train_indices].pin_memory().to(device, non_blocking=True),hidden_states = None,exit_at = -1)
    labels = batch_data['labels'][selected_train_indices].pin_memory().to(device, non_blocking=True)
    loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
    loss *= batch_data['mask_ids'][selected_train_indices].view(-1).pin_memory().to(device, non_blocking=True)
    loss = loss.mean()/batch_data['mask_ids'][selected_train_indices].view(-1).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    sched.step()

    to_log.update({'train/optimized_loss':loss, 'train/lr':optimizer.param_groups[-1]['lr']})
    
    # if (to_log.get('train/objective_loss') is None):
    #     to_log['train/objective_loss'] = loss

    if (iter_%100 == 0 and master_process):
        wandb.log(to_log)

    if (iter_ % args.evaluation_every_kiterations == 0):
        with torch.no_grad():
            g_cpu = torch.Generator()
            g_cpu.manual_seed(100)
            test_losses,test_accuracies = [],[]
            for index_ in range(0,test_size,args.test_batch_size):
                batch_indices = np.arange(args.test_batch_size)+index_
                batch_data = data_collator([tokenized_dataset['test'][int(i)]['input_ids'][:args.max_seq_len] for i in batch_indices])
                lm_logits = model(batch_data['input_ids'].pin_memory().to(device, non_blocking=True),hidden_states = None,exit_at = -1)
                batch_data['mask_ids'] = torch.bernoulli(torch.zeros(batch_data['input_ids'].shape)+0.15,generator=g_cpu)
                batch_data['input_ids'][batch_data['mask_ids'] == 1] = 103


                labels = batch_data['labels'].pin_memory().to(device, non_blocking=True)
                loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
                loss *= batch_data['mask_ids'].view(-1).pin_memory().to(device, non_blocking=True)
                loss = loss.mean()/batch_data['mask_ids'].view(-1).mean()

                acc = (lm_logits.view(-1, lm_logits.size(-1)).argmax(axis=-1) == labels.view(-1)).float().mean()
                test_losses.append(float(loss))
                test_accuracies.append(float(acc))
    
            if (master_process):
                wandb.log({'test/loss':np.array(test_losses).mean(),'test/acc':np.array(test_accuracies).mean()},commit=False)


if (args.save and master_process):
    torch.save(model.state_dict(),args.save_ckpt_name)

if (args.ddp):
    destroy_process_group()
