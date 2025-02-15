from tinygrad.nn.optim import Adam
from tinygrad.tensor import Tensor
import time
import numpy as np
from model import SRCNN  # Import the model
from data import create_dataset

def train(model, lr_tensor, hr_tensor, learning_rate=1e-4, num_epochs=10, batch_size=64):
    """Trains the SRCNN model."""

    optimizer = Adam(model.parameters(), lr=learning_rate)

    with Tensor.train(): # VERY IMPORTANT: Enable training mode
      for epoch in range(num_epochs):
        epoch_loss = 0.0
        start_time = time.time()

        # Iterate in batches
        for i in range(0, lr_tensor.shape[0], batch_size):
            batch_lr = lr_tensor[i:min(i + batch_size, lr_tensor.shape[0])]
            batch_hr = hr_tensor[i:min(i + batch_size, hr_tensor.shape[0])]

            optimizer.zero_grad()
            output = model(batch_lr)
            loss = ((output - batch_hr)**2).mean()  # MSE Loss

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_time = time.time() - start_time
        avg_loss = epoch_loss / (lr_tensor.shape[0] // batch_size)  # More accurate loss reporting
        print(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.6f}, Time: {epoch_time:.2f}s")


if __name__ == '__main__':
    # Example usage
    upscale_factor = 3
    model = SRCNN(upscale_factor=upscale_factor)

    #Replace with paths to a few images for testing (at least two)
    image_paths = ["./data/archive/dataset/train/high_res/1.png", "./data/archive/dataset/train/high_res/2.png"]

    lr_tensor, hr_tensor = create_dataset(image_paths, upscale_factor)
    train(model, lr_tensor, hr_tensor, learning_rate=1e-4, num_epochs=20, batch_size=64)