# Plot the loss values
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

import numpy as np

loss_values = np.loadtxt("../model/losses.csv", delimiter=",", usecols=0)
print(loss_values)

psnr_values = np.loadtxt("../results/evaluation_metrics_results.csv", usecols=1, skiprows=1, delimiter=",")
print(psnr_values)

ssim_values = np.loadtxt("../results/evaluation_metrics_results.csv", usecols=2, skiprows=1, delimiter=",")
print(ssim_values)

num_epochs = len(loss_values) #// 296
loss_x = np.arange(0, num_epochs + 1, 1 / 296)
loss_x = loss_x[:len(loss_values)]

num_epochs = len(psnr_values) #// 3
psnr_x = np.arange(0, num_epochs + 1, 1 )#/ 3)
psnr_x = psnr_x[:len(psnr_values)]

num_epochs = len(ssim_values) #// 3
ssim_x = np.arange(0, num_epochs + 1, 1 )#/ 3)
ssim_x = ssim_x[:len(ssim_values)]

# Plot the loss values
plt.figure(figsize=(8, 6))
plt.plot(loss_x, loss_values, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.savefig('../results/training_loss.png')
plt.show()

# Plot the PSNR values
plt.figure(figsize=(8, 6))
plt.plot(psnr_x, psnr_values, label='PSNR')
plt.xlabel('Epoch')
plt.ylabel('PSNR')
plt.title('Validation PSNR')
plt.legend()
plt.savefig('../results/validation_psnr.png')
plt.show()

# Plot the SSIM values
plt.figure(figsize=(8, 6))
plt.plot(ssim_x, ssim_values, label='SSIM')
plt.xlabel('Epoch')
plt.ylabel('SSIM')
plt.title('Validation SSIM')
plt.legend()
plt.savefig('../results/validation_ssim.png')
plt.show()
