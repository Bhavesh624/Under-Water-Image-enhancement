import os
import time
import torch
import torch.nn as nn
import numpy as np
from utils import get_image, imsave_label
from torchvision import models

class T_CNN(nn.Module):
    def __init__(self, device, image_height=460, image_width=620,
                 label_height=460, label_width=620, batch_size=1,
                 c_dim=3, checkpoint_dir=None, sample_dir=None):
        super(T_CNN, self).__init__()
        self.device = device
        self.image_height = image_height
        self.image_width = image_width
        self.label_height = label_height
        self.label_width = label_width
        self.batch_size = batch_size
        self.c_dim = c_dim
        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir
        


        self.main_branch = nn.Sequential(
            nn.Conv2d(12, 128, 7, padding=3), nn.ReLU(),
            nn.Conv2d(128, 128, 5, padding=2), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 64, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 7, padding=3), nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=2), nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1), nn.Sigmoid()
        )

        def small_branch():
            return nn.Sequential(
                nn.Conv2d(6, 32, 7, padding=3), nn.ReLU(),
                nn.Conv2d(32, 32, 5, padding=2), nn.ReLU(),
                nn.Conv2d(32, 3, 3, padding=1), nn.ReLU()
            )

        self.branch_wb = small_branch()
        self.branch_ce = small_branch()
        self.branch_gc = small_branch()

    def forward(self, image, image_wb, image_ce, image_gc):
        concat_all = torch.cat([image, image_wb, image_ce, image_gc], dim=1)
        weights = self.main_branch(concat_all)
        weight_wb, weight_ce, weight_gc = torch.chunk(weights, 3, dim=1)

        wb1 = self.branch_wb(torch.cat([image, image_wb], dim=1))
        ce1 = self.branch_ce(torch.cat([image, image_ce], dim=1))
        gc1 = self.branch_gc(torch.cat([image, image_gc], dim=1))

        output = wb1 * weight_wb + ce1 * weight_ce + gc1 * weight_gc
        return output

    def test_image(self, config, test_paths, id):
        image_test = get_image(test_paths[0], is_grayscale=False)
        wb_test = get_image(test_paths[1], is_grayscale=False)
        ce_test = get_image(test_paths[2], is_grayscale=False)
        gc_test = get_image(test_paths[3], is_grayscale=False)

        image_tensor = torch.tensor(image_test.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        wb_tensor = torch.tensor(wb_test.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        ce_tensor = torch.tensor(ce_test.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        gc_tensor = torch.tensor(gc_test.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

        self.eval()
        with torch.no_grad():
            start_time = time.time()
            output = self.forward(image_tensor, wb_tensor, ce_tensor, gc_tensor)
            elapsed = time.time() - start_time
            print(f"Processing time: {elapsed:.4f}s")

            output_image = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
            output_image = np.clip(output_image, 0, 1)
            save_dir = os.path.join(os.getcwd(), config.sample_dir)
            os.makedirs(save_dir, exist_ok=True)
            output_path = os.path.join(save_dir, os.path.basename(test_paths[0]))
            imsave_label(output_image, output_path)
            print(f"Saved: {output_path}")

    def load_weights(self, weight_path):
        checkpoint = torch.load(weight_path, map_location=self.device)
        self.load_state_dict(checkpoint)
        print("[*] Model loaded successfully.")