import cv2
import numpy as np
import os
from tinygrad.tensor import Tensor
from config import Config

def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0

def generate_patches(hr_img, scale, patch_size):
    # Generate low-res and bicubic upscaled images
    lr_img = cv2.resize(hr_img, (hr_img.shape[1]//scale, hr_img.shape[0]//scale), 
                       interpolation=cv2.INTER_CUBIC)
    lr_up = cv2.resize(lr_img, (hr_img.shape[1], hr_img.shape[0]), 
                      interpolation=cv2.INTER_CUBIC)
    
    # Generate random crop
    h, w = lr_up.shape[:2]
    x = np.random.randint(0, h - patch_size)
    y = np.random.randint(0, w - patch_size)
    
    lr_patch = lr_up[x:x+patch_size, y:y+patch_size]
    hr_patch = hr_img[x:x+patch_size, y:y+patch_size]
    
    # Convert to CHW format
    return (lr_patch.transpose(2, 0, 1),
            hr_patch.transpose(2, 0, 1))

class SRCNNDataset:
    def __init__(self, hr_folder, scale):
        self.hr_images = [os.path.join(hr_folder, f) 
                         for f in os.listdir(hr_folder) if f.endswith(('png', 'jpg'))]
        self.scale = scale

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, idx):
        hr = load_image(self.hr_images[idx])
        lr, hr = generate_patches(hr, self.scale, Config.patch_size)
        return Tensor(lr), Tensor(hr)