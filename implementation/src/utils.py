import argparse


def load_config(config_path):

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"ile not found: {config_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON file: {config_path} - {e}")    

def parse_args():
    parser = argparse.ArgumentParser(description='TUW-ML_UE3: SRCNN Parameters')
    parser.add_argument('--config', type=str, help='Path to the config file', default='../src/config.json')
    parser.add_argument('--vis_num_images', type=int, help='Numer of comparison images/visualizations')
    parser.add_argument('--vis_save_path', type=str, help='Directory path to save comparison images/visualizations')
    parser.add_argument('--max_vis', type=int, help='Maximum number of images/visualizations')
    parser.add_argument('--eval_score_path', type=str, help='Directory path to save evaluation scores file')
    
    parser.add_argument('--resample_scale_factor', type=int, help='Loss image quality for input: resample_scale_factor')
    parser.add_argument('--batch-size', type=int, help='The batch size to use during training')
    parser.add_argument('--batch-size-test', type=int, help='The batch size to use during testing')
    
    parser.add_argument('--learning_rate', type=float, help='Learning rate for the optimizer')
    parser.add_argument('--epochs', type=int, help='Number of epochs to train the model')
    
    parser.add_argument('--download_input_path', type=str, help='Download folder for the images')

    args = parser.parse_args()
    return args