import argparse

import torch
import torch.optim as optim
import yaml
from torch import nn
from tqdm import tqdm

from data_loading import get_dataloaders
from model import SRCNN
from test import evaluate_model
import os 
from utils import load_config, parse_args

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
losses = list()


def train_model(model, trainloader, validloader, criterion, optimizer, epochs, config):
    print(torch.cuda.is_available())
    model = model.to(device)
    model.train()

    # Early stopping params
    best_score = float('-inf')
    counter = 0
    patience = 10

    log_loss_every_nth_batch = len(trainloader) // 10
    validate_every_nth_batch = len(trainloader) // 3
   
    for epoch in range(1, epochs + 1):
        tqdm_postfix = {'loss': 0}
        accumulated_loss = 0
        tqdm_loader = tqdm(trainloader)
        tqdm_loader.set_description(f"Epoch [{epoch}/{epochs}]")

        for i, data in enumerate(tqdm_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # add statistics to progress bar
            tqdm_postfix['loss'] = loss.item()
            tqdm_loader.set_postfix(tqdm_postfix)
            losses.append(loss.item())
            if i % log_loss_every_nth_batch == log_loss_every_nth_batch - 1:
                print(f"Average loss of current batch interval: {accumulated_loss / log_loss_every_nth_batch:.4f}")
                accumulated_loss = 0
            if i % validate_every_nth_batch == validate_every_nth_batch - 1:
                validation_metrics = evaluate_model(model, validloader, config)
                model.train()

                print(validation_metrics)
                if validation_metrics['PSNR'] > best_score:
                    best_score = validation_metrics['PSNR']
                    torch.save(model.state_dict(), f'../model/model_{epoch}_{i * 3 // len(trainloader)}.pt')
                    counter = 0
                else:
                    print("No further improvement")
                    counter += 1

                if counter >= patience:
                    print("Early stopping")
                    return model

        torch.save(model.state_dict(), f'../model/model_{epoch}_final.pt')


        
    return model


if __name__ == '__main__':

    args = parse_args()

    # Load configuration from file
    config = load_config(args.config)

    # Override config with command-line arguments if provided
    if args.learning_rate is not None:
        config['train']['learning_rate'] = args.learning_rate
    if args.epochs is not None:
        config['train']['epochs'] = args.epochs

    learning_rate = config['train']['learning_rate']
    epochs = config['train']['epochs']

    model = SRCNN()
    #model.load_state_dict(torch.load('../model/model_32_final.pt', map_location=device))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    optimizer2 = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, dampening=0.2, weight_decay=1e-5)
    #optimizer3 = optim.LBFGS(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    trainloader, validloader, _ = get_dataloaders(seed=42)

    model = train_model(model, trainloader, validloader, criterion, optimizer2, epochs, config)

    torch.save(model.state_dict(), '../model/model.pt')
    filename = os.path.join("../model/", 'losses.csv')
    with open(filename, "w") as file:
        for loss_value in losses:
            file.write(str(loss_value)+"\n")
