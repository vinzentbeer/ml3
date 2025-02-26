import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


from data_loader import get_dataloaders
from model import SRCNN
from test import evaluate_model
from utils import load_config, parse_args
import logging

#constants and logger
MODEL_DIR = "../model"
LOSSES_FILE = os.path.join(MODEL_DIR, 'losses.csv')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#helper functions (but not nice and functional, they have side effects :/)

def save_model(model, epoch, step=None, is_final=False):
    filename = f"model_{epoch}"
    if step is not None:
        filename += f"_{step}"
    filename += ".pt"
    if(is_final):
        filename = "model.pt"
    filepath = os.path.join(MODEL_DIR, filename)
    
    torch.save(model.state_dict(), filepath)
    logging.info(f"Model saved to {filepath}")


def save_losses(losses, filename=LOSSES_FILE):
    with open(filename, "a") as file:
        for loss_value in losses:
            file.write(str(loss_value) + "\n")  # Added newline for CSV format
    logging.info(f"Losses saved to {filename}")


def create_optimizer(model, config):
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
    model.train()  #set model to training mode!!!! This is weirdly named, since it sounds like it would train straight away
    
    log_loss_every_n_batches = config['train'].get('log_loss_every_n_batches', len(trainloader) // 10)
    # Ensure at least one log per epoch, even if trainloader is small
    log_loss_every_n_batches = max(1, log_loss_every_n_batches)

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
        losses.append(str(loss.item()) + ", Epoch " + str(epoch))

        tqdm_loader.set_postfix({'loss': loss.item()})  # Update progress bar

        if (i + 1) % log_loss_every_n_batches == 0:
            avg_loss = accumulated_loss / log_loss_every_n_batches
            logging.info(f"Epoch [{epoch}/{config['train']['epochs']}], Batch [{i+1}/{len(trainloader)}], Avg Loss: {avg_loss:.4f}")
            accumulated_loss = 0.0

    return losses


def validate_model(model, validloader, criterion, device, config):
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

    model = model.to(device)

    epochs = config['train']['epochs']
    validate_every_n_epochs = config['train'].get('validate_every_n_epochs', 1)  # Validate every epoch by default
    patience = config['train'].get('patience', 10)  # Default to 10 if patience not in config

    best_score = float('-inf')
    counter = 0  #for early stopping. if it goes over patience, we stop
    losses = []  

    for epoch in range(1, epochs + 1):
        losses = train_one_epoch(model, trainloader, criterion, optimizer, epoch, device, config, losses)

        # Validation and Early Stopping
        if epoch % validate_every_n_epochs == 0:
            validation_metrics = validate_model(model, validloader, criterion, device, config)

            if validation_metrics['PSNR'] > best_score:
                best_score = validation_metrics['PSNR']
                save_model(model, epoch, step="best")
                counter = 0  # Reset early stopping counter. This logic is not that good, it almost always fires even when it might not be the best
            else:
                logging.info("No improvement in PSNR.")
                counter += 1
                if counter >= patience:
                    logging.info("Early stopping triggered.")
                    break  # Exit training loop here

        save_model(model, epoch, step="final")  #saves a model per epoch. might be overkill.

    save_losses(losses)
    return model




# --- Main Execution ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SRCNN Training Script")
    parser.add_argument('--config', type=str, default='config.json', help='Path to the config file.')
    parser.add_argument('--learning_rate', type=float, help='Learning rate for training.')
    parser.add_argument('--epochs', type=int, help='Number of epochs to train.')

    args = parse_args()
    config = load_config(args.config)

    # Override config with command-line arguments if provided
    if args.learning_rate is not None:
        config['train']['learning_rate'] = args.learning_rate
    if args.epochs is not None:
        config['train']['epochs'] = args.epochs

    resample_scale_factor = args.resample_scale_factor if args.resample_scale_factor is not None else config['input']['resample_scale_factor']
    batch_size = args.batch_size if args.batch_size is not None else config['train']['batch_size']
    batch_size_test = args.batch_size_test if args.batch_size_test is not None else config['evaluate']['batch_size']
    subset_percentage = config['dataset'].get('subset_percentage', 0.1) #Default to 0.1 if not set
    dataset_folder = config['dataset'].get('dataset_folder', 'train2017') 


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    logging.info(f"Using dataset folder: {dataset_folder}")
    logging.info(f"Using subset percentage: {subset_percentage}")
    logging.info(f"Using resample scale factor: {resample_scale_factor}")

    MODEL_DIR = config['train']['model_dir'] #does this work with constants?
    LOSSES_FILE = os.path.join(MODEL_DIR, 'losses.csv')

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR, exist_ok=True)


    model = SRCNN().to(device) #Move model to device here
    start_from_checkpoint = config['train'].get('start_from_checkpoint', False)
    if(start_from_checkpoint):
        model.load_state_dict(torch.load('../model/model_50_final.pt', map_location=device
    ))  
    #model.load_state_dict(torch.load('../model/model_39_final.pt', map_location=device))
    criterion = nn.MSELoss()
    optimizer = create_optimizer(model, config) # Use function to create optimizer
    #dataset_folder = config["dataset"].get("dataset_folder", "train2017")
    




    trainloader, validloader, _ = get_dataloaders(seed=42, subset_percentage=subset_percentage, resample_scale_factor=resample_scale_factor,batch_size=batch_size, batch_size_test=batch_size_test, dataset_folder=dataset_folder)

    model = train_model(model, trainloader, validloader, criterion, optimizer, config, device)

    #save the final model as just model.pt
    save_model(model, config['train']['epochs'], step="final", is_final=True)
    #oved save_losses call into the train_model function after training completes.
    logging.info("Training complete! Woo! Yeah! We did it! We trained a model! Yay!")