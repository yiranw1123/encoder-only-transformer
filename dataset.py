from tokenizers import Tokenizer, normalizers, pre_tokenizers
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from pathlib import Path
from config import get_config
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
import torch
from tqdm.auto import tqdm

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
        encoded_sentence = self.df.dataset.iloc[index]['tokenized']
        num_padding_tokens = self.max_length - len(encoded_sentence) - 2


        
        padded_encoded_tokens = torch.cat([
            self.sos_token,
            torch.tensor(encoded_sentence, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] *num_padding_tokens, dtype=torch.int64)
        ])

        assert padded_encoded_tokens.size(0) == self.max_length

        return{
            'index': index,
            'sentence': sentence,
            'encoder_input':padded_encoded_tokens,
            'attention_mask':(padded_encoded_tokens != self.pad_token).unsqueeze(0).unsqueeze(0).int(), #(1, 1, seq_len)
            'target': torch.tensor(target, dtype = torch.long)
        }

def read_raw_files(config):
    train = process_file(config['train_file_name'], type = 'train', max_token_len=config['max_seq_len'])
    test = process_file(config['test_file_name'], type = 'test',  max_token_len=config['max_seq_len'])
    return train, test


def write_to_file(lines, type, n_lines):
    cnt = 1
    with open(f'./data/{type}_{cnt}.txt', 'w', encoding='utf-8') as f:
       line_cnt = 0
       i = 0
       while i < len(lines) and line_cnt <n_lines:
           f.write(lines[i][0])
           f.write('\t')
           f.write(lines[i+1][0])
           f.write('\n')
           i+=2
           line_cnt += 1
    f.close()


def create_df(train, test):
    train_df = pd.DataFrame(train, columns= ['sentence', 'target'])
    # add a column in train to display class label
    class_names = {1: 'correct', 0: 'incorrect'}
    train_df['label'] = train_df['target'].map(class_names)

    test_df = pd.DataFrame(test, columns=['sentence'])
    return train_df, test_df

def filter_by_max_seq_len(df, tokenizer, config):
    token_lens = []
    tokenized = []

    for txt in df.sentence:
        tokens = tokenizer.encode(txt, add_special_tokens = True)
        tokenized.append(tokens.ids)
        token_lens.append(len(tokens))
    
    df['token_lens'] = token_lens
    df['tokenized'] = tokenized

    filtered_df = df.loc[df['token_lens'] <= config['max_seq_len'] - 2]
    return filtered_df


def prepare_dataset(config):
    train, test = read_raw_files(config)
    train_df, test_df = create_df(train, test)

    # build tokenizer from train data
    train_tokenizer = get_or_build_tokenizer(config, train_df)
    train_df = filter_by_max_seq_len(train_df, train_tokenizer, config)

    train_df = train_df.iloc[:10000]

    # train - val split
    val_ds_size = int(config['validation_size'] * len(train_df))
    train_ds_size = len(train_df) - val_ds_size
    train_ds_raw, val_ds_raw = random_split(train_df, [train_ds_size, val_ds_size])

    train_ds = EnglishSentenceDataSet(train_ds_raw, train_tokenizer, config['max_seq_len'])
    val_ds = EnglishSentenceDataSet(val_ds_raw, train_tokenizer, config['max_seq_len'])
    test_ds = EnglishSentenceDataSet(test_df, train_tokenizer, config['max_seq_len'] )

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_ds, batch_size = config['batch_size'], shuffle= True)

    return train_dataloader, val_dataloader, test_dataloader, train_tokenizer


def process_file(file_name, type: str, max_token_len: int):
    res = []
    with open(file_name, 'r', encoding='utf-8', errors='replace') as file:
        lines = file.readlines()
        for line in lines:
            fields = line.strip().split('\t')
            if type == 'test' :
                sent1, sent2 = fields
                if(len(sent1.split()) <= max_token_len):
                    res.append(sent1)
                    res.append(sent2)
            elif type == 'train':
                sent1, sent2 = fields
                if(len(sent1.split()) <= max_token_len):
                    res.append([sent1, 1])
                    res.append([sent2, 0])
    
    max_record = 10000 if type=='train' else 1000
    return res


def get_all_sentences(df):
    for sentence in df['sentence']:
        yield sentence

def get_or_build_tokenizer(config, dataset):
    tokenizer_path = Path(config['tokenizer_file'])
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordPiece(unk_token = "[UNK]"))
        tokenizer.normalizer = normalizers.Sequence([normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()])
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        trainer = WordPieceTrainer(special_tokens = ["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency = 2)
        tokenizer.train_from_iterator(get_all_sentences(dataset), trainer = trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


if __name__ == "__main__":
    config = get_config()
    tqdm(prepare_dataset(config))

