import os


def create_project(name: str):
    # Create the directory for the project
    save_path = os.path.join("logs", name)
    if os.path.isdir(save_path):
        raise FileExistsError("Run Already Exists. Don't overwrite your data")

    os.makedirs(save_path)
    return save_path


def add_to_project(module: str, parent: str):
    # Create Module 
    save_path = os.path.join(parent, module)
    if os.path.isdir(save_path):
        raise FileExistsError("Run Already Exists. Don't overwrite your data")
    os.makedirs(save_path)

    return save_path
