import argparse
import os
import urllib.request

import tqdm
import yaml

from utils import load_config, parse_args

args = parse_args()

# Load configuration from file
config = load_config(args.config)

# Override config with command-line arguments if provided
if args.download_input_path is not None:
        config['input']['download_input_path'] = args.download_input_path
download_input_path = config['input']['download_input_path']

file_path = download_input_path + '/train2017'

if os.path.exists(file_path):
    print(f"Training data 'train2017' folder already exists at path: {download_input_path}")
    print("If you want to download the dataset again, please delete the folder first.")
    exit(0)

# download the coco dataset from http://images.cocodataset.org/zips/train2017.zip

url = 'http://images.cocodataset.org/zips/train2017.zip'
# add a progress bar to the download
with urllib.request.urlopen(url) as response:
    with open('../data/train2017.zip', 'wb') as out_file:
        length = response.info()['Content-Length']
        length = int(length)
        chunk_size = 16 * 1024
        with tqdm.tqdm(total=length, unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                pbar.update(len(chunk))

# unzip the dataset
import zipfile

with zipfile.ZipFile('../data/train2017.zip', 'r') as zip_ref:
    zip_ref.extractall('../data')
