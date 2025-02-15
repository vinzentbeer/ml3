# this file creates the dataloaders for the dataset
import argparse
import os

import torch
import yaml
from PIL import Image
from PIL.Image import Resampling
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms


from utils import load_config, parse_args

args = parse_args()

# Load configuration from file
config = load_config(args.config)

# Override config with command-line arguments if provided
if args.resample_scale_factor is not None:
    config['input']['resample_scale_factor'] = args.resample_scale_factor
if args.batch_size is not None:
    config['train']['batch_size'] = args.batch_size
if args.batch_size_test is not None:
    config['evaluate']['batch_size'] = args.batch_size_test

resample_scale_factor = config['input']['resample_scale_factor']
batch_size = config['train']['batch_size']
batch_size_test = config['evaluate']['batch_size']

# https://web.archive.org/web/20230628113759/https://yizhepku.github.io/2020/12/26/dataloader.html
class CustomDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.original_dir = os.path.join(root_dir, 'train2017')

        # List images in the directories
        self.first_10_percent_images = os.listdir(self.original_dir)
        self.first_10_percent_images = self.first_10_percent_images[:int(len(self.first_10_percent_images) * 0.1)]

    def __len__(self):
        return len(self.first_10_percent_images)

    def __getitem__(self, idx):
        original_img_path = os.path.join(self.original_dir, self.first_10_percent_images[idx])
        original_image = Image.open(original_img_path).convert('RGB')

        # Loss image quality for input
        width, height = original_image.size
        scale_factor = resample_scale_factor
        new_size = (width // scale_factor, height // scale_factor)
        resized_image = original_image.resize(new_size, resample=Resampling.BICUBIC)
        resized_image = resized_image.resize(original_image.size, resample=Resampling.BICUBIC)

        return resized_image, original_image


def pad_images_to_same_size(batch):
    # Find maximum dimensions
    max_width = max([img.size[0] for img, label in batch])
    max_height = max([img.size[1] for img, label in batch])

    # Resize and pad images
    padded_imgs = []
    for img, label in batch:
        pad_width = max_width - img.size[0]
        pad_height = max_height - img.size[1]
        pad_left = pad_width // 2
        pad_top = pad_height // 2
        pad_right = pad_width - pad_left
        pad_bottom = pad_height - pad_top

        padded_img = transforms.Pad((pad_left, pad_top, pad_right, pad_bottom), fill=255)(img)
        padded_label = transforms.Pad((pad_left, pad_top, pad_right, pad_bottom), fill=255)(label)
        padded_imgs.append((padded_img, padded_label))

    # Stack images and labels
    images = torch.stack([transforms.ToTensor()(img) for img, _ in padded_imgs])
    labels = torch.stack([transforms.ToTensor()(label) for _, label in padded_imgs])

    return images, labels


def get_dataloaders(seed: int = None):
    dataset = CustomDataset('../data/')

    generator = torch.Generator()
    if seed is not None:
        generator = generator.manual_seed(seed)
    train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [0.8, 0.1, 0.1],
        generator=generator
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        #num_workers=3,
        collate_fn=pad_images_to_same_size,
        pin_memory=True,
        shuffle=True
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size_test,
        #num_workers=3,
        collate_fn=pad_images_to_same_size,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size_test,
        #num_workers=3,
        collate_fn=pad_images_to_same_size,
        pin_memory=True
    )

    return train_loader, validation_loader, test_loader
