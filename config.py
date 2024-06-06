from pathlib import Path

def get_config():
    return {
        "train_file_name":'./data/subtrain.txt',
        "test_file_name": './data/subtest.txt',
        
        # training params
        "validation_size": 0.2,
        "batch_size": 8,
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


        "model_folder": "weights",
        "model_basename": "tmodel_",
        "tokenizer_file": "tokenizer.json",
        "experiment_name": "runs/tmodel"
    }