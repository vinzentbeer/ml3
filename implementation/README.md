# Topic 3.2: Deep Learning: Image Upscaling

## Setup

Run the following commands to install dependencies.  
You can also use your IDE to achieve this.

pip install uv
python3 -m uv sync 

this will create a venv you can use to run the code.

Alternatively, if you don't want to use uv, set up a python 3.10 installation and run pip install -r requirements.txt


Commands to download, train and test the model:

```shell
cd src
python3 get_dataset.py
python3 train.py
python3 test.py
```

## Results

The trained models can be found in `model/`.  
There are snapshots for each epoch and a final model.

| Model    | PSNR    | SSIM   |
|----------|---------|--------|
| model.pt | 28.9451 | 0.9194 |

### Visualizations

In the following our result after 100 epochs of training is displayed. 
In this case it is not the coco dataset but another smaller one we also used to compare.

![visualization_1.png](visualizations/2025-02-25%2012_07_44/visualization_1.png)

Here the coco dataset results are shown:



### Dataset

The coco dataset was used to train the model. We used 10% of the dataset 
even though it was very large. (more than 12h training time)

### Training


![loss.png](results/loss.png)
![psnr.png](results/psnr.png)
![ssim.png](results/ssim.png)
