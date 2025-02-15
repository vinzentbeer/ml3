class Config:
    # Training parameters
    scale_factor = 3          # Scaling factor for super-resolution
    batch_size = 16           # Training batch size
    epochs = 100              # Number of training epochs
    lr = 0.001                # Learning rate
    patch_size = 33           # Size of training patches
    
    # Dataset paths
    train_hr_path = "./data/archive/dataset/train/high_res/"
    valid_hr_path = "./data/archive/dataset/val/high_res/"