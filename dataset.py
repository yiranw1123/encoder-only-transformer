from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
from config import get_config
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
import torch

class EnglishSentenceDataSet(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.sos_token = torch.tensor([tokenizer.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer.token_to_id("[PAD]")], dtype=torch.int64)


    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        sentence = self.df.dataset.iloc[index]['sentence']
        target = self.df.dataset.iloc[index]['target']
        encoded_sentence = self.tokenizer.encode(sentence).ids
        num_padding_tokens = self.max_length - len(encoded_sentence) - 2

        if num_padding_tokens < 0 :
            raise ValueError("Sentence is too long")
        
        padded_encoded_tokens = torch.cat([
            self.sos_token,
            torch.tensor(encoded_sentence, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] *num_padding_tokens, dtype=torch.int64)
        ])

        assert padded_encoded_tokens.size(0) == self.max_length

        return{
            'encoder_input':padded_encoded_tokens,
            'attention_mask':(padded_encoded_tokens != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            'target': torch.tensor(target, dtype = torch.long)
        }



def prepare_dataset(config):
    train = process_file(config['train_file_name'], type = 'train')
    test = process_file(config['test_file_name'], type = 'test')
    train_df = pd.DataFrame(train, columns= ['sentence', 'target'])
    test_df = pd.DataFrame(test, columns=['sentence'])
    # build tokenizer from train data
    train_tokenizer = get_or_build_tokenizer(config, train_df)

    # Keep 90% for training, 10% for validation
    train_ds_size = int(0.9 * len(train_df))
    val_ds_size = len(train_df) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(train_df, [train_ds_size, val_ds_size])

    train_ds = EnglishSentenceDataSet(train_ds_raw, train_tokenizer, config['max_seq_len'])
    val_ds = EnglishSentenceDataSet(val_ds_raw, train_tokenizer, config['max_seq_len'])

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, train_tokenizer


def process_file(file_name, type: str):
    res = []
    with open(file_name, 'r', encoding='utf-8', errors='replace') as file:
        lines = file.readlines()
        for line in lines:
            fields = line.strip().split('\t')
            if type == 'test' :
                res.append(fields[0])
                res.append(fields[1])
            elif type == 'train':
                res.append([fields[0], 1])
                res.append([fields[1], 0])
    return res

def get_all_sentences(df):
    for sentence in df['sentence']:
        yield sentence


def get_or_build_tokenizer(config, dataset):
    tokenizer_path = Path(config['tokenizer_file'])
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token = "[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(dataset), trainer = trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


if __name__ == "__main__":
    config = get_config()
    prepare_dataset(config)

