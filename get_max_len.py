from config import get_config
import time

config = get_config()

file_name = config['train_file_name']

sentences = []
with open(file_name, 'r', encoding='utf-8', errors='replace') as file:
    lines = file.readlines()
    num_sentences = 0
    for line in lines:
        sentence, _ = line.strip().split('\t')
        sentences.append(sentence)
        num_sentences += 1
    

def get_max_seq_length(sentences):
    res = [len(sentence.split()) for sentence in sentences]
    return max(res)
    
start_time = time.time()
max_lengths = get_max_seq_length(sentences)
end_time = time.time()

print(f"Processed {num_sentences} sentences in {end_time - start_time} seconds")
print(f"Max length: {max_lengths}")
## max length is 668