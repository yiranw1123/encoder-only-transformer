from dataset import prepare_dataset
from config import get_config
from model import TransformerForSentenceClassification
import torch
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from tqdm import tqdm
import numpy as np

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
            
            
            outputs = model.forward(inputs)
            valid_loss += criterion(outputs, targets)
            num_batches += 1

            # Get predictions
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

        # Optionally, print out accuracy or other metrics
        accuracy = (np.array(all_predictions) == np.array(all_targets)).mean()
        print(f'Validation Accuracy: {accuracy * 100:.2f}%')


            

# Function to construct the path for saving and retrieving model weights
def get_weights_file_path(config, epoch: str):
    model_folder = config['model_folder'] # Extracting model folder from the config
    model_basename = config['model_basename'] # Extracting the base name for model files
    model_filename = f"{model_basename}{epoch}.pt" # Building filename
    return str(Path('.')/ model_folder/ model_filename) # Combining current directory, the model folder, and the model filename


def train_model(config):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") 
    print(f"Using device: {device}")
    
    train_dataloader, val_dataloader, tokenizer = prepare_dataset(config=config)
    vocab_size = tokenizer.get_vocab_size()
    # Creating model directory to store weights
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    model = TransformerForSentenceClassification(config, vocab_size).to(device)

    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])
    
    # Setting up the Adam optimizer with the specified learning rate from the '
    # config' dictionary plus an epsilon value
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps = 1e-9)
    
    # Initializing epoch and global step variables
    initial_epoch = 0
    global_step = 0
    
    # Initializing CrossEntropyLoss function for training
    # We ignore padding tokens when computing loss, as they are not relevant for the learning process
    # We also apply label_smoothing to prevent overfitting
    criterion = nn.CrossEntropyLoss(ignore_index = tokenizer.token_to_id('[PAD]'), label_smoothing = 0.1).to(device)

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
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['target'].to(device)
            
            # Running tensors through the Transformer
            output = model.forward(inputs)
            
            # Computing loss between model's output and true labels
            loss = criterion(output, targets)
            
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
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
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