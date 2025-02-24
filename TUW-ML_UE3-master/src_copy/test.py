import argparse
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, MultiScaleStructuralSimilarityIndexMeasure, PeakSignalNoiseRatioWithBlockedEffect
from torchvision.utils import make_grid
from tqdm import tqdm

from data_loader import get_dataloaders
from model import SRCNN

from utils import load_config, parse_args

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_model(model, testloader, config):

    model = model.to(device)
    #model.eval() switches layers like BatchNorm and Dropout to evaluation mode
    model.eval()
    
    eval_score_path = config['evaluate']['eval_score_path']
    # get metrics like in the paper
    psnr = PeakSignalNoiseRatio().to(device)
    ssim = StructuralSimilarityIndexMeasure().to(device)
    msssim = MultiScaleStructuralSimilarityIndexMeasure().to(device)
    #psnrbe = PeakSignalNoiseRatioWithBlockedEffect().to(device)

    total_psnr = 0.0
    total_ssim = 0.0
    total_msssim = 0.0
    #total_psnrbe = 0.0
    num_batches = 0

    #no grad: disables gradient calculation during model inference or evaluation since we are not training. 
    # since our tensors are initialized with requires_grad=true by default, we need to explicitly disable it here
    with torch.no_grad():
        for i, data in enumerate(tqdm(testloader)):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            # Accumulate PSNR and SSIM values
            total_psnr += psnr(outputs, labels).item()
            total_ssim += ssim(outputs, labels).item()
            total_msssim += msssim(outputs, labels).item()
            #total_psnrbe += psnrbe(outputs, labels).item()
            num_batches += 1

        
        psnr_ret = round(total_psnr / num_batches, 4)
        ssim_ret = round(total_ssim / num_batches, 4)
        msssim_ret = round(total_msssim / num_batches, 4)
        #psnrbe_ret = round(total_psnrbe / num_batches, 4)

        # also write evaluation scores
        directory = eval_score_path
        if not os.path.exists(directory):
            os.makedirs(directory)

        filename = os.path.join(directory, 'evaluation_scores.csv')

        # Check if the file exists, create with headers if not
        if not os.path.exists(filename):
            with open(filename, "w") as file:
                file.write("Date/Time, PSNR, SSIM\n")

        # Open the file in append mode
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(filename, "a") as file:
            file.write(f'{current_time}, {psnr_ret}, {ssim_ret}')
            file.write('\n')

    return {
        'PSNR': psnr_ret,
        'SSIM': ssim_ret,
        'MSSSIM': msssim_ret,
        #'PSNRBE': psnrbe_ret,
    }


def visualize(model, testloader, config):
    model = model.to(device)
    model.eval()
    
    
    num_images = config['visualize']['vis_num_images']
    save_path = config['visualize']['vis_save_path']
    max_visualizations = config['visualize']['max_vis']


    save_path = save_path + '/' + datetime.now().strftime('%Y-%m-%d %H_%M_%S')

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    visualization_count = 0

    with torch.no_grad():
        for i, data in enumerate(tqdm(testloader)):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            # Convert the tensors to RGB images
            inputs_rgb = inputs.cpu().numpy()
            labels_rgb = labels.cpu().numpy()
            outputs_rgb = outputs.cpu().numpy()

            # Combine the images into a grid
            img_grid = np.concatenate((inputs_rgb[:num_images], labels_rgb[:num_images], outputs_rgb[:num_images]),
                                      axis=0)

            # Create a figure and display the grid
            fig, ax = plt.subplots(figsize=(20, 20))
            ax.imshow(make_grid(torch.from_numpy(img_grid), nrow=num_images).permute(1, 2, 0))
            ax.axis('off')
            plt.title('Input Images (Downscaled), Ground Truth, and Model Outputs')

            # Save the plot
            plt.savefig(os.path.join(save_path, f'visualization_{i + 1}.png'))

            visualization_count += 1
            if visualization_count >= max_visualizations:
                break


if __name__ == '__main__':
    
    args = parse_args()
        
    # Load configuration from file
    config = load_config(args.config)
    
    ## Override config with command-line arguments if provided
    #vis_num_images = args.vis_num_images if args.vis_num_images is not None else config['visualize']['vis_num_images']
    #vis_save_path = args.vis_save_path if args.vis_save_path is not None else config['visualize']['vis_save_path']
    #max_vis = args.max_vis if args.max_vis is not None else config['visualize']['max_vis']
    #eval_score_path = args.eval_score_path if args.eval_score_path is not None else config['evaluate']['eval_score_path']
    
    # Override config with command-line arguments if provided
    if args.vis_num_images is not None:
        config['visualize']['vis_num_images'] = args.vis_num_images
    if args.vis_save_path is not None:
        config['visualize']['vis_save_path'] = args.vis_save_path
    if args.max_vis is not None:
        config['visualize']['max_vis'] = args.max_vis
    if args.eval_score_path is not None:
        config['evaluate']['eval_score_path'] = args.eval_score_path



    model = SRCNN()
    model.load_state_dict(torch.load('../model/model.pt', map_location=device))
    _, _, testloader = get_dataloaders(seed=42)
    
    visualize(model, testloader, config)
    metrics = evaluate_model(model, testloader, config)

    for k, v in metrics.items():
        print(f'{k}: {v}')
