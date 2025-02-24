import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from data_loader import get_dataloaders
from model import SRCNN
from test import evaluate_model
from utils import load_config, parse_args
import logging

# --- Constants ---
MODEL_DIR = "../model"
LOSSES_FILE = os.path.join(MODEL_DIR, 'losses.csv')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions ---

def save_model(model, epoch, step=None):
    """Saves the model state dictionary."""
    filename = f"model_{epoch}"
    if step is not None:
        filename += f"_{step}"
    filename += ".pt"
    filepath = os.path.join(MODEL_DIR, filename)
    torch.save(model.state_dict(), filepath)
    logging.info(f"Model saved to {filepath}")


def save_losses(losses, filename=LOSSES_FILE):
    """Saves the list of losses to a CSV file."""
    with open(filename, "w") as file:
        for loss_value in losses:
            file.write(str(loss_value) + "\n")  # Added newline for CSV format
    logging.info(f"Losses saved to {filename}")


def create_optimizer(model, config):
    """Creates an optimizer based on the configuration."""
    optimizer_config = config['train']['optimizer']
    optimizer_name = optimizer_config['name'].lower()
    learning_rate = config['train']['learning_rate']
    weight_decay = config['train'].get('weight_decay', 0) #default to 0 if not set
    
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        momentum = optimizer_config.get('momentum', 0)  #Default to 0 if not provided
        dampening = optimizer_config.get('dampening', 0) #Default to 0 if not provided
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, dampening=dampening, weight_decay=weight_decay)
    #elif optimizer_name == 'lbfgs':  # Not generally recommended for deep learning
    #    optimizer = optim.LBFGS(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    return optimizer



def train_one_epoch(model, trainloader, criterion, optimizer, epoch, device, config, losses):
    """Trains the model for one epoch."""
    model.train()  # Set model to training mode
    
    log_loss_every_nth_batch = config['train'].get('log_loss_every_nth_batch', len(trainloader) // 10) #Get the number of logging intervals from the config, if not present, use default
    if log_loss_every_nth_batch == 0:
        log_loss_every_nth_batch = 1 # log every batch if there are few batches

    accumulated_loss = 0.0
    tqdm_loader = tqdm(trainloader, desc=f"Epoch [{epoch}/{config['train']['epochs']}]")

    for i, (inputs, labels) in enumerate(tqdm_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()  # Clear gradients
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        # --- Logging and Statistics ---
        accumulated_loss += loss.item()
        losses.append(loss.item())

        tqdm_loader.set_postfix({'loss': loss.item()})  # Update progress bar

        if (i + 1) % log_loss_every_nth_batch == 0:
            avg_loss = accumulated_loss / log_loss_every_nth_batch
            logging.info(f"Epoch [{epoch}/{config['train']['epochs']}], Batch [{i+1}/{len(trainloader)}], Avg Loss: {avg_loss:.4f}")
            accumulated_loss = 0.0

    return losses


def validate_model(model, validloader, criterion, device, config):
    """Evaluates the model on the validation set."""
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    with torch.no_grad():  # Disable gradient calculation during validation
        for inputs, labels in validloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)  # Accumulate loss

    avg_loss = total_loss / len(validloader.dataset)
    logging.info(f"Validation Loss: {avg_loss:.4f}")

    validation_metrics = evaluate_model(model, validloader, config)
    logging.info(validation_metrics)
    return validation_metrics

def train_model(model, trainloader, validloader, criterion, optimizer, config, device):
    """
    Trains the SRCNN model with validation and early stopping.

    Args:
        model: The SRCNN model.
        trainloader: DataLoader for the training set.
        validloader: DataLoader for the validation set.
        criterion: Loss function.
        optimizer: Optimizer.
        config: Configuration dictionary.
        device:  'cuda' or 'cpu'

    Returns:
        The trained model.
    """
    model = model.to(device)
    
    epochs = config['train']['epochs']
    validate_every_nth_batch = config['train'].get('validate_every_nth_batch', len(trainloader) // 3) #Get the number of validation intervals from the config, if not present, use default

    patience = config['train'].get('patience', 10) # Default to 10 if patience not in config
    
    best_score = float('-inf')
    counter = 0  # Early stopping counter
    losses = []  # Store training losses
    
    for epoch in range(1, epochs + 1):
        losses = train_one_epoch(model, trainloader, criterion, optimizer, epoch, device, config, losses)

        # Validation and Early Stopping
        if validate_every_nth_batch > 0 and epoch % (validate_every_nth_batch // (len(trainloader) // 10) if (len(trainloader) // 10) else 1) == 0:  #Approximate original validation frequency
            validation_metrics = validate_model(model, validloader, criterion, device, config)
            
            if validation_metrics['PSNR'] > best_score:
                best_score = validation_metrics['PSNR']
                save_model(model, epoch, step="best")
                counter = 0  # Reset early stopping counter
            else:
                logging.info("No improvement in PSNR.")
                counter += 1
                if counter >= patience:
                    logging.info("Early stopping triggered.")
                    break  # Exit training loop

        save_model(model, epoch, step="final") #Save at the end of each epoch

    save_losses(losses) # Save losses to file after training
    return model




# --- Main Execution ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SRCNN Training Script")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file.')
    parser.add_argument('--learning_rate', type=float, help='Learning rate for training.')
    parser.add_argument('--epochs', type=int, help='Number of epochs to train.')

    args = parse_args()
    config = load_config(args.config)

    # Override config with command-line arguments if provided
    if args.learning_rate is not None:
        config['train']['learning_rate'] = args.learning_rate
    if args.epochs is not None:
        config['train']['epochs'] = args.epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    model = SRCNN().to(device) #Move model to device here

    criterion = nn.MSELoss()
    optimizer = create_optimizer(model, config) # Use function to create optimizer


    trainloader, validloader, _ = get_dataloaders(seed=42)

    model = train_model(model, trainloader, validloader, criterion, optimizer, config, device)


    save_model(model, config['train']['epochs'], step="final")
    # Moved save_losses call into the train_model function after training completes.
    logging.info("Training complete.")