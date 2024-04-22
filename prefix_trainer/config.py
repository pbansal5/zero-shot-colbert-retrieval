from dataclasses import dataclass

@dataclass
class TrainingConfig:
    learning_rate_schedule: str
    learning_rate_schedule_params: dict
    epochs: int
    batch_size: int
    loss: str
    save_embedding_rate: int
    optimizer: str
    optimizer_params: dict
    layer: int
