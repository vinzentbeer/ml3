import argparse
import os
import torch
import PIL
from PIL import Image#, Resampling
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from typing import Tuple, Optional, List
from utils import load_config, parse_args
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Monkey-patch PIL.Image.Resampling for Pillow<9.0
if not hasattr(PIL.Image, 'Resampling'):  # Pillow<9.0
    Resampling = PIL.Image
class SRCNNDataset(Dataset):
    """
    Dataset for SRCNN super-resolution training.  Loads pairs of low-resolution
    (degraded) and high-resolution (original) images.
    """

    def __init__(self, root_dir: str, resample_scale_factor: int = 2, subset_percentage: float = 0.1, dataset_dir:str = 'train2017'):
       
        self.root_dir = root_dir
        self.original_dir = os.path.join(root_dir, dataset_dir)
        self.resample_scale_factor = resample_scale_factor

        # Load a subset of images for faster iteration
        image_files = os.listdir(self.original_dir)
        num_images = int(len(image_files) * subset_percentage)
        self.image_files = image_files[:num_images]
        logging.info(f"Using {len(self.image_files)} images from {self.original_dir}")


    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            A tuple containing:
                - low_res_image (torch.Tensor): The degraded, low-resolution image.
                - high_res_image (torch.Tensor): The original, high-resolution image.
        """
        img_name = self.image_files[idx]
        img_path = os.path.join(self.original_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        #create low res image based on scale factors and convert to tensors
        width, height = image.size
        new_size = (width // self.resample_scale_factor, height // self.resample_scale_factor)
        low_res_image = image.resize(new_size, resample=PIL.Image.BICUBIC)
        low_res_image = low_res_image.resize(image.size, resample=PIL.Image.BICUBIC)  # Upscale back

        transform = transforms.ToTensor()
        low_res_tensor = transform(low_res_image)
        high_res_tensor = transform(image)

        return low_res_tensor, high_res_tensor


def pad_images_to_same_size(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:

    max_height = max([img.shape[1] for img, label in batch])  # Shape is (C, H, W)
    max_width = max([img.shape[2] for img, label in batch])

    padded_imgs = []
    padded_labels = []
    for img, label in batch:
        pad_width = max_width - img.shape[2]
        pad_height = max_height - img.shape[1]
        pad_left = pad_width // 2
        pad_top = pad_height // 2
        pad_right = pad_width - pad_left
        pad_bottom = pad_height - pad_top

        # Use functional padding for tensors
        padded_img = torch.nn.functional.pad(img, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=1)  # Assuming 1 is white
        padded_label = torch.nn.functional.pad(label, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=1)

        padded_imgs.append(padded_img)
        padded_labels.append(padded_label)

    # Stack images and labels
    images = torch.stack(padded_imgs)
    labels = torch.stack(padded_labels)

    return images, labels


def get_dataloaders(root_dir: str = '../data/',
                    resample_scale_factor: int = 2,
                    batch_size: int = 32,
                    batch_size_test: int = 64,
                    subset_percentage: float = 0.1,
                    seed: Optional[int] = None,
                    val_split: float = 0.1,
                    test_split: float = 0.1,
                    dataset_folder:str = "train2017") -> Tuple[DataLoader, DataLoader, DataLoader]:
 

    dataset = SRCNNDataset(root_dir, resample_scale_factor, subset_percentage, dataset_folder)
    total_len = len(dataset)
    val_len = int(val_split * total_len)
    test_len = int(test_split * total_len)
    train_len = total_len - val_len - test_len

    generator = torch.Generator()
    if seed is not None:
        generator = generator.manual_seed(seed)

    train_dataset, validation_dataset, test_dataset = random_split(
        dataset,
        [train_len, val_len, test_len],
        generator=generator
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=pad_images_to_same_size,
        pin_memory=True,
        #num_workers = os.cpu_count() # Use all CPUs available
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size_test,
        shuffle=False,  # No need to shuffle validation data
        collate_fn=pad_images_to_same_size,
        pin_memory=True,
        #num_workers = os.cpu_count()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size_test,
        shuffle=False,
        collate_fn=pad_images_to_same_size,
        pin_memory=True,
        #num_workers = os.cpu_count()
    )

    logging.info(f"Train dataset size: {len(train_dataset)}")
    logging.info(f"Validation dataset size: {len(validation_dataset)}")
    logging.info(f"Test dataset size: {len(test_dataset)}")

    return train_loader, validation_loader, test_loader



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SRCNN DataLoaders")
    parser.add_argument('--config', type=str, default='config.json', help='Path to the config file.')
    parser.add_argument('--resample_scale_factor', type=int, help='Resample scale factor.')
    parser.add_argument('--batch_size', type=int, help='Batch size for training.')
    parser.add_argument('--batch_size_test', type=int, help='Batch size for testing.')
    args = parser.parse_args()

    config = load_config(args.config)

    resample_scale_factor = config['input']['resample_scale_factor']
    batch_size = config['train']['batch_size']
    batch_size_test =  config['evaluate']['batch_size']
    subset_percentage = config['dataset'].get('subset_percentage', 0.1) #Default to 0.1 if not set
    dataset_folder = config['dataset'].get('dataset_folder', 'train2017') #Default to 'train2017' if not set



    train_loader, val_loader, test_loader = get_dataloaders(
        root_dir='../data/',
        resample_scale_factor=resample_scale_factor,
        batch_size=batch_size,
        batch_size_test=batch_size_test,
        subset_percentage=subset_percentage,
        seed=42, 
        dataset_folder=dataset_folder
    )

    #debug output
    for i, (inputs, labels) in enumerate(train_loader):
        print("Input batch shape:", inputs.shape)
        print("Label batch shape:", labels.shape)
        break