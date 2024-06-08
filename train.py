from dataset import prepare_dataset
from config import get_config, latest_weights_file_path
import torch
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from bertviz.neuron_view import show
from model import Encoder, InputEmbedding, MultiHeadAttentionBlock, FeedForwardBlock, EncoderBlock, PositionalEncoding, SentenceClassificationTransformer, ClassificationLayer
def initialize_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0, std=1)


def build_model(config, vocab_size):

    d_model = config['d_model']
    num_labels = config['num_labels']
    max_seq_len = config['max_seq_len']
    dropout = config['hidden_dropout_prob']
    d_ff = config['d_ff']
    h = config['num_attention_heads']
    num_layers = config['num_hidden_layers']


    src_embd = InputEmbedding(d_model, vocab_size)
    src_pos_embd = PositionalEncoding(d_model, max_seq_len)

    encoder_blocks = []

    for _ in range(num_layers):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))

    classifier = ClassificationLayer(d_model, num_labels)

    model = SentenceClassificationTransformer(encoder, src_embd, src_pos_embd, classifier)

    # Initialize the parameters
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return model


# Defining function to evaluate the model on the validation dataset
# num_examples = 2, two examples per run
def run_validation(model, validation_ds, criterion, tokenizer, device, print_msg, global_state, writer, num_examples=2):
    model.eval() # Setting model to evaluation mode
    console_width = 80 # Fixed witdh for printed messages
    
    # Creating evaluation loop
    with torch.no_grad():
        valid_loss = 0.0
        num_batches = 0 # Initializing counter to keep track of how many examples have been processed
        all_predictions = []
        all_targets = []
        
        # Ensuring that no gradients are computed during this process
        for batch in validation_ds:
            
            inputs = batch['encoder_input'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['target'].to(device)
            
            
            outputs = model.forward(inputs, attention_mask)
            valid_loss += criterion(outputs, targets)
            num_batches += 1

            # Get predictions
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

        # Optionally, print out accuracy or other metrics
        accuracy = (np.array(all_predictions) == np.array(all_targets)).mean()
        print(f'Validation Accuracy: {accuracy * 100:.2f}%')

def train_model(config):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") 
    print(f"Using device: {device}")
    
    train_dataloader, val_dataloader, tokenizer = prepare_dataset(config=config)
    vocab_size = tokenizer.get_vocab_size()
    # Creating model directory to store weights
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    model = build_model(config, vocab_size).to(device)
    model.apply(initialize_weights)

    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])
    
    # Setting up the Adam optimizer with the specified learning rate from the '
    # config' dictionary plus an epsilon value
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps = 1e-9)
    
    # Initializing epoch and global step variables
    initial_epoch = 0
    global_step = 0

    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')
    
    # Initializing CrossEntropyLoss function for training
    # We ignore padding tokens when computing loss, as they are not relevant for the learning process
    # We also apply label_smoothing to prevent overfitting
    criterion = nn.CrossEntropyLoss().to(device)

        # Iterating over each epoch from the 'initial_epoch' variable up to
    # the number of epochs informed in the config
    for epoch in range(initial_epoch, config['num_epochs']):
        
        # Initializing an iterator over the training dataloader
        # We also use tqdm to display a progress bar
        batch_iterator = tqdm(train_dataloader, desc = f'Processing epoch {epoch:02d}')
        
        # For each batch...
        for batch in batch_iterator:
            model.train() # Train the model
            
            # Loading input data and masks onto the GPU
            inputs = batch['encoder_input'].to(device)
            src_mask = batch['attention_mask'].to(device)
            targets = batch['target'].to(device)

            if torch.isnan(inputs).any():
                print('invalid input detected at iteration ', i)
            
            # Running tensors through the Transformer
            output = model.forward(inputs, src_mask)
            
            # Computing loss between model's output and true labels
            loss = criterion(output, targets)

            if torch.isnan(loss):
                print("NaN loss encountered")
                continue  # Skip this batch
            
            # Updating progress bar
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})
            
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()
            
            # Performing backpropagation
            loss.backward()
            
            # Updating parameters based on the gradients
            optimizer.step()
            
            # Clearing the gradients to prepare for the next batch
            optimizer.zero_grad()
            
            global_step += 1 # Updating global step count
            
        # We run the 'run_validation' function at the end of each epoch
        # to evaluate model performance
        run_validation(model, val_dataloader, criterion, tokenizer, device, lambda msg: batch_iterator.write(msg), global_step, writer)
         
        # Saving model
        model_filename = get_weights_file_path(config, f'{epoch+1:02d}')
        # Writting current model state to the 'model_filename'
        torch.save({
            'epoch': epoch, # Current epoch
            'model_state_dict': model.state_dict(),# Current model state
            'optimizer_state_dict': optimizer.state_dict(), # Current optimizer state
            'global_step': global_step # Current global step 
        }, model_filename)

        epoch += 1
    


if __name__ == '__main__':
    config = get_config()
    train_model(config)