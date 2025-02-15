import time
import numpy as np
from tinygrad.nn.optim import Adam
from model import SRCNN
from data_loader import SRCNNDataset
from config import Config
from tinygrad.tensor import Tensor, dtypes


def train():
    # Initialize dataset and model
    train_dataset = SRCNNDataset(Config.train_hr_path, Config.scale_factor)
    model = SRCNN()
    opt = Adam(model.parameters(), lr=Config.lr)

    # Training loop
    for epoch in range(Config.epochs):
        start = time.time()
        losses = []
        
        # Shuffle dataset
        indices = np.random.permutation(len(train_dataset))
        
        for i in range(0, len(indices), Config.batch_size):
            batch_indices = indices[i:i+Config.batch_size]
            
            """# Get batch
            inputs, targets = [], []
            for idx in batch_indices:
                lr, hr = train_dataset[idx]
                inputs.append(lr)
                targets.append(hr)"""
            # Create tensors directly from numpy arrays
            batch_data = [train_dataset[idx] for idx in batch_indices]
            # Stack the numpy arrays first, then create a single tensor
            input_array = np.stack([lr for lr, _ in batch_data])
            target_array = np.stack([hr for _, hr in batch_data])
            
            # Convert to tensors with proper dtype
            inputs = Tensor(input_array, dtype=dtypes.float32)
            targets = Tensor(target_array, dtype=dtypes.float32)
                
            # Forward pass
            opt.zero_grad()
            output = model(Tensor.stack(inputs))
            loss = (output - Tensor.stack(targets)).square().mean()
            
            # Backward pass
            loss.backward()
            opt.step()
            
            losses.append(loss.numpy())
            
        # Print epoch statistics
        avg_loss = np.mean(losses)
        print(f"Epoch {epoch+1}/{Config.epochs} | Loss: {avg_loss:.4f} | "
              f"Time: {time.time()-start:.2f}s")

if __name__ == "__main__":
    train()