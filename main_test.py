import os
import glob
import argparse
import pprint
import numpy as np
import torch
from model_train import T_CNN
from utils import get_image

pp = pprint.PrettyPrinter()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--image_height", type=int, default=112)
    parser.add_argument("--image_width", type=int, default=112)
    parser.add_argument("--label_height", type=int, default=112)
    parser.add_argument("--label_width", type=int, default=112)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--c_dim", type=int, default=3)
    parser.add_argument("--c_depth_dim", type=int, default=1)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoint")
    parser.add_argument("--sample_dir", type=str, default="sample")
    parser.add_argument("--test_data_dir", type=str, default="test")
    parser.add_argument("--is_train", action='store_true')
    args = parser.parse_args()

    pp.pprint(vars(args))

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)

    def get_data_paths(folder):
        data_dir = os.path.join(os.getcwd(), folder)
        paths = []
        for ext in ["*.png", "*.jpg", "*.bmp", "*.jpeg"]:
            paths.extend(glob.glob(os.path.join(data_dir, ext)))
        return sorted(paths)

    test_data_list = get_data_paths('test_real')
    test_data_list1 = get_data_paths('input_wb_test')
    test_data_list2 = get_data_paths('input_ce_test')
    test_data_list3 = get_data_paths('input_gc_test')
    print(test_data_list)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    min_len = min(len(test_data_list), len(test_data_list1), len(test_data_list2), len(test_data_list3))
    print(f"Using {min_len} samples for inference...")

    for ide in range(min_len):
        image_test = get_image(test_data_list[ide], is_grayscale=False)
        wb_test = get_image(test_data_list1[ide], is_grayscale=False)
        ce_test = get_image(test_data_list2[ide], is_grayscale=False)
        gc_test = get_image(test_data_list3[ide], is_grayscale=False)

        shape = image_test.shape
        shape = image_test.shape
        target_height, target_width = shape[0], shape[1]

        import cv2
        wb_test = cv2.resize(wb_test, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
        ce_test = cv2.resize(ce_test, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
        gc_test = cv2.resize(gc_test, (target_width, target_height), interpolation=cv2.INTER_CUBIC)

        model = T_CNN(
            device=device,
            image_height=shape[0],
            image_width=shape[1],
            label_height=args.label_height,
            label_width=args.label_width,
            batch_size=args.batch_size,
            c_dim=args.c_dim,
            checkpoint_dir=args.checkpoint_dir,
            sample_dir=args.sample_dir
        ).to(device)

        image_tensor = torch.tensor(image_test.transpose(2, 0, 1)).unsqueeze(0).to(device)
        wb_tensor = torch.tensor(wb_test.transpose(2, 0, 1)).unsqueeze(0).to(device)
        ce_tensor = torch.tensor(ce_test.transpose(2, 0, 1)).unsqueeze(0).to(device)
        gc_tensor = torch.tensor(gc_test.transpose(2, 0, 1)).unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            output = model(image_tensor, wb_tensor, ce_tensor, gc_tensor)
            output_image = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
            output_image = (output_image * 255).astype(np.uint8)
            output_path = os.path.join(args.sample_dir, f"output_{ide}.png")
            from imageio import imwrite
            imwrite(output_path, output_image)
            print(f"Saved: {output_path}")

if __name__ == '__main__':
    main()