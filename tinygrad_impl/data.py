import numpy as np
from PIL import Image
import cv2
from tinygrad.tensor import Tensor

def load_and_preprocess(image_path, upscale_factor, sub_image_size=33, stride=14):
    """Loads, preprocesses, and creates sub-images."""

    img = Image.open(image_path).convert('L')  # Grayscale
    img_array = np.array(img).astype(np.float32) / 255.0

    # Simulate LR image (blur, downsample, upscale)
    lr_img_array = blur_and_downsample(img_array, upscale_factor)
    lr_img = Image.fromarray((lr_img_array * 255).astype(np.uint8))
    # Use BICUBIC interpolation for upscaling, as in the paper.
    lr_img = lr_img.resize((img.width, img.height), Image.BICUBIC)
    lr_img_array = np.array(lr_img).astype(np.float32) / 255.0

    lr_sub_images, hr_sub_images = [], []
    for y in range(0, img.height - sub_image_size + 1, stride):
        for x in range(0, img.width - sub_image_size + 1, stride):
            lr_sub = lr_img_array[y:y+sub_image_size, x:x+sub_image_size]
            hr_sub = img_array[y:y+sub_image_size, x:x+sub_image_size]

            lr_sub_images.append(lr_sub.reshape(1, 1, sub_image_size, sub_image_size))  # C, H, W
            hr_sub_images.append(hr_sub.reshape(1, 1, sub_image_size, sub_image_size))

    return lr_sub_images, hr_sub_images  # Return lists of sub-images

def blur_and_downsample(img_array, upscale_factor):
    """Simple blurring and downsampling (replace with better kernel)."""
    from scipy.ndimage import gaussian_filter
    blurred = gaussian_filter(img_array, sigma=0.5 + upscale_factor/6)  # Adjust sigma.  Experiment!
    lr_height, lr_width = img_array.shape[0] // upscale_factor, img_array.shape[1] // upscale_factor
    return resize_numpy_image(blurred, (lr_width, lr_height))


def resize_numpy_image(image:np.ndarray, size:tuple[int, int]) -> np.ndarray:
  """resizes numpy arrays using cv2"""
  return cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)

def create_dataset(image_paths, upscale_factor, sub_image_size=33, stride=14):
    """Creates a dataset of sub-images from multiple images."""
    all_lr_sub_images, all_hr_sub_images = [], []
    for image_path in image_paths:
        lr_subs, hr_subs = load_and_preprocess(image_path, upscale_factor, sub_image_size, stride)
        all_lr_sub_images.extend(lr_subs)
        all_hr_sub_images.extend(hr_subs)

    # Convert to NumPy arrays and then to Tensors
    all_lr_sub_images = np.concatenate(all_lr_sub_images, axis=0)
    all_hr_sub_images = np.concatenate(all_hr_sub_images, axis=0)

    return Tensor(all_lr_sub_images), Tensor(all_hr_sub_images)

if __name__ == '__main__':
    # Example Usage (replace with your data)
    image_paths = ["path/to/image1.png", "path/to/image2.png"] #use multiple images
    upscale_factor = 3
    lr_tensor, hr_tensor = create_dataset(image_paths, upscale_factor)
    print(f"LR Tensor shape: {lr_tensor.shape}, HR Tensor shape: {hr_tensor.shape}")