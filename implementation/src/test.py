import argparse
import os
from datetime import datetime
import logging
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

def evaluate_model(model, testloader, config, log_file="train_eval_metrics.csv"):

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

            # Accumulates metrics
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

        filename = os.path.join(directory, log_file)

        
        if not os.path.exists(filename):
            with open(filename, "w") as file:
                file.write("Date/Time, PSNR, SSIM, MSSSIM\n")

        # Open the file in append mode
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(filename, "a") as file:
            file.write(f'{current_time}, {psnr_ret}, {ssim_ret}, {msssim_ret}')
            file.write('\n')

    return {
        'PSNR': psnr_ret,
        'SSIM': ssim_ret,
        'MSSSIM': msssim_ret,
        #'PSNRBE': psnrbe_ret,
    }
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def visualize(model, testloader, config):
    """
    Visualizes the model's output by creating a grid of input (downscaled),
    ground truth, and model output images, with labels for each row.

    Args:
        model: The SRCNN model.
        testloader: DataLoader for the test set.
        config: Configuration dictionary containing visualization settings.
        device: The device (CPU or GPU) to use.
    """
    model.to(device)
    model.eval()

    num_images = config['visualize']['vis_num_images']
    save_path = config['visualize']['vis_save_path']
    max_visualizations = config['visualize']['max_vis']
    start_at = 6
    save_path = os.path.join(save_path, datetime.now().strftime('%Y-%m-%d_%H_%M_%S'))
    os.makedirs(save_path, exist_ok=True)
    #num_images = num_images + start_at
    visualization_count = 0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(testloader, desc="Visualizing")):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            #number of images to visualize from the batch
            inputs_vis = inputs[:num_images].cpu()
            labels_vis = labels[:num_images].cpu()
            outputs_vis = outputs[:num_images].cpu()

            #make sure all tensors have the same number of channels (3 for RGB).... nasty bug potential
            if inputs_vis.shape[1] != 3:
                inputs_vis = inputs_vis.repeat(1, 3 // inputs_vis.shape[1], 1, 1)
            if labels_vis.shape[1] != 3:
                labels_vis = labels_vis.repeat(1, 3 // labels_vis.shape[1], 1, 1)
            if outputs_vis.shape[1] != 3:
                outputs_vis = outputs_vis.repeat(1, 3 // outputs_vis.shape[1], 1, 1)

            # Create a grid for each row
            input_grid = make_grid(inputs_vis, nrow=num_images, padding=2, pad_value=1) #pad_value for white background
            label_grid = make_grid(labels_vis, nrow=num_images, padding=2, pad_value=1)
            output_grid = make_grid(outputs_vis, nrow=num_images, padding=2, pad_value=1)

            # Convert the grids to numpy arrays and transpose dimensions for display
            input_grid_np = input_grid.cpu().permute(1, 2, 0).numpy()
            label_grid_np = label_grid.cpu().permute(1, 2, 0).numpy()
            output_grid_np = output_grid.cpu().permute(1, 2, 0).numpy()

            # Clip values to be within the valid range [0, 1]
            input_grid_np = np.clip(input_grid_np, 0, 1)
            label_grid_np = np.clip(label_grid_np, 0, 1)
            output_grid_np = np.clip(output_grid_np, 0, 1)

         
            fig, axs = plt.subplots(3, 1, figsize=(15, 15))  # 3 rows, 1 column

            # Display each grid with a label
            axs[0].imshow(input_grid_np)
            axs[0].set_title("Input (Downscaled)")
            axs[0].axis('off')

            axs[1].imshow(label_grid_np)
            axs[1].set_title("Ground Truth")
            axs[1].axis('off')

            axs[2].imshow(output_grid_np)
            axs[2].set_title("Model Output")
            axs[2].axis('off')

            # Adjust layout to prevent overlapping titles
            plt.tight_layout()

            # Save the plot
            filepath = os.path.join(save_path, f'visualization_{i + 1}.png')
            plt.savefig(filepath)
            logging.info(f"Visualization saved to {filepath}")
            plt.close(fig)

            visualization_count += 1
            if visualization_count >= max_visualizations:
                break


if __name__ == '__main__':
    
    args = parse_args()
        
    config = load_config(args.config)
  
    
    # Override config with command-line arguments if provided
    if args.vis_num_images is not None:
        config['visualize']['vis_num_images'] = args.vis_num_images
    if args.vis_save_path is not None:
        config['visualize']['vis_save_path'] = args.vis_save_path
    if args.max_vis is not None:
        config['visualize']['max_vis'] = args.max_vis
    if args.eval_score_path is not None:
        config['evaluate']['eval_score_path'] = args.eval_score_path


  
    resample_scale_factor = config["input"]["resample_scale_factor"]





    model = SRCNN()
    model.load_state_dict(torch.load('../model_ratrainedWith5EpochsWithMoreData/model_55_best.pt', map_location=device))
    _, _, testloader = get_dataloaders(seed=42, resample_scale_factor=resample_scale_factor)
    
    visualize(model, testloader, config)
    metrics = evaluate_model(model, testloader, config)

    for k, v in metrics.items():
        print(f'{k}: {v}')
