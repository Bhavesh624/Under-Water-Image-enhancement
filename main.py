import os
import argparse
import pprint
import torch
from model_train import T_CNN
from utils import prepare_data

pp = pprint.PrettyPrinter()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=400, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--image_height", type=int, default=112, help="Image height")
    parser.add_argument("--image_width", type=int, default=112, help="Image width")
    parser.add_argument("--label_height", type=int, default=112, help="Label height")
    parser.add_argument("--label_width", type=int, default=112, help="Label width")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--beta1", type=float, default=0.5, help="Momentum term of Adam")
    parser.add_argument("--c_dim", type=int, default=3, help="Color dimension")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoint", help="Checkpoint directory")
    parser.add_argument("--sample_dir", type=str, default="sample", help="Sample directory")
    parser.add_argument("--test_data_dir", type=str, default="test", help="Test data directory")
    parser.add_argument("--is_train", action='store_true', help="Set to training mode")
    args = parser.parse_args()

    pp.pprint(vars(args))

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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