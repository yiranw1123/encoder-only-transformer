from pathlib import Path

def get_config():
    return {
        "train_file_name":'./data/train-1.txt',
        "test_file_name": './data/subtest.txt',
        
        # training params
        "validation_size": 0.2,
        "batch_size": 16,
        "num_epochs": 2,
        "lr": 10**-4,

        #model params
        "num_hidden_layers": 12,
        "max_seq_len": 50,
        'd_ff': 3072, # feed-forward layer size
        "d_model": 768,
        "num_attention_heads": 12,
        'num_labels':2,
        "hidden_dropout_prob": 0.1,

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