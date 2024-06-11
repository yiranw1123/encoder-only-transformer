from pathlib import Path

def get_config():
    return {
        "train_file_name":'./data/train_1.txt',
        "test_file_name": './data/subtest.txt',
        
        # training params
        "validation_size": 0.2,
        "batch_size": 8,
        "num_epochs": 2,
        "lr": 0.0005,

        #model params
        "n_layers": 3,
        "max_seq_len": 57,
        'd_ff': 128, # feed-forward layer size
        "d_model": 512,
        "n_head": 8,
        'num_labels':2,
        "dropout": 0.1,

        "datasource": "checkpoints",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "tokenizer_file": "tokenizer.json",
        "experiment_name": "runs/tmodel",
        "preload": "latest"
    }


def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])