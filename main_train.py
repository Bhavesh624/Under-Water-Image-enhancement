import torch
import os
import pprint
import argparse
import matplotlib.pyplot as plt
import numpy as np

from model_train import T_CNN
from utils import imsave, prepare_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=400, help="Number of epochs [default: 400]")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size [default: 16]")
    parser.add_argument("--image_height", type=int, default=112, help="Image height")
    parser.add_argument("--image_width", type=int, default=112, help="Image width")
    parser.add_argument("--label_height", type=int, default=112, help="Label height")
    parser.add_argument("--label_width", type=int, default=112, help="Label width")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--beta1", type=float, default=0.5, help="Momentum term of Adam")
    parser.add_argument("--c_dim", type=int, default=3, help="Number of color channels")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoint", help="Checkpoint directory")
    parser.add_argument("--sample_dir", type=str, default="sample", help="Sample directory")
    parser.add_argument("--test_data_dir", type=str, default="test", help="Test data directory")
    parser.add_argument("--is_train", type=bool, default=True, help="True for training, False for testing")

    args = parser.parse_args()
    pp = pprint.PrettyPrinter()
    pp.pprint(vars(args))

    # Create directories if not exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)

    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize and train model
    model = T_CNN(
        device=device,
        image_height=args.image_height,
        image_width=args.image_width,
        label_height=args.label_height,
        label_width=args.label_width,
        batch_size=args.batch_size,
        c_dim=args.c_dim,
        checkpoint_dir=args.checkpoint_dir,
        sample_dir=args.sample_dir
    )

    model.train_model(args)

if __name__ == '__main__':
    main()
