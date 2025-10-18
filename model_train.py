import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from utils import imsave, prepare_data, get_image
import time
import matplotlib.pyplot as plt
import random

class T_CNN(nn.Module):
    def __init__(self,
                 device,
                 image_height=230,
                 image_width=310,
                 label_height=230,
                 label_width=310,
                 batch_size=2,
                 c_dim=3,
                 checkpoint_dir=None,
                 sample_dir=None):
        super(T_CNN, self).__init__()
        self.device = device
        self.image_height = image_height
        self.image_width = image_width
        self.label_height = label_height
        self.label_width = label_width
        self.batch_size = batch_size
        self.c_dim = c_dim
        self.dropout_keep_prob = 0.5

        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir
        self.CONTENT_LAYER = 'relu5_4'
        self.a = nn.Parameter(torch.ones(1),requires_grad=True)
        self.b = nn.Parameter(torch.ones(1),requires_grad=True)
        self.c = nn.Parameter(torch.ones(1),requires_grad=True)
        self.d = nn.Parameter(torch.ones(1),requires_grad=True)
       


        self.main_branch = nn.Sequential(
            nn.Conv2d(c_dim * 4, 128, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        def small_branch():
            return nn.Sequential(
                nn.Conv2d(c_dim * 2, 32, kernel_size=7, padding=3),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Conv2d(32, 3, kernel_size=3, padding=1),
                nn.ReLU()
            )

        self.branch_wb = small_branch()
        self.branch_ce = small_branch()
        self.branch_gc = small_branch()

        vgg19 = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()
        self.vgg_layers = vgg19[:36]
        for param in self.vgg_layers.parameters():
            param.requires_grad = False

    def forward(self, image, image_wb, image_ce, image_gc):
        concat_all = torch.cat([image, image_wb, image_ce, image_gc], dim=1)
        weights = self.main_branch(concat_all)
        weight_wb, weight_ce, weight_gc = torch.chunk(weights, 3, dim=1)

        wb1 = self.branch_wb(torch.cat([image, image_wb], dim=1))
        ce1 = self.branch_ce(torch.cat([image, image_ce], dim=1))
        gc1 = self.branch_gc(torch.cat([image, image_gc], dim=1))

        output = wb1 * weight_wb + ce1 * weight_ce + gc1 * weight_gc
        return output * self.b

    def extract_vgg_features(self, x):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        x = (x - mean) / std
        features = x
        for i, layer in enumerate(self.vgg_layers):
            features = layer(features)
            if i == 35:
                break
        return features

    def compute_loss(self, pred, label):
        # l1_loss = torch.exp(-self.a) * F.l1_loss(pred, label) + self.a
        l1_loss =  torch.exp(-self.a) * F.l1_loss(pred, label) 
        pred_vgg = self.extract_vgg_features(pred)
        label_vgg = self.extract_vgg_features(label)
        # vgg_loss = torch.exp(-self.c) * F.mse_loss(pred_vgg, label_vgg) +self.c 
        vgg_loss = torch.exp(-self.c) * F.mse_loss(pred_vgg, label_vgg) 
        # return torch.exp(-self.a) * (0.05 * vgg_loss + l1_loss) + self.a
        return torch.exp(-self.d) * (vgg_loss + l1_loss) 

    def train_model(self, config):
        random.seed(42)
        best_psnr = 0.0
        best_ssim = 0.0
        best_epoch = -1
        if config.is_train:
            data_train_list = prepare_data("input_train")
            data_wb_train_list = prepare_data("input_wb_train")
            data_ce_train_list = prepare_data("input_ce_train")
            data_gc_train_list = prepare_data("input_gc_train")
            image_train_list = prepare_data("gt_train")

        data_test_list = prepare_data("input_test")
        data_wb_test_list = prepare_data("input_wb_test")
        data_ce_test_list = prepare_data("input_ce_test")
        data_gc_test_list = prepare_data("input_gc_test")
        image_test_list = prepare_data("gt_test")

        self.to(self.device)
        optimizer = torch.optim.Adam(self.parameters(), lr=config.learning_rate, betas=(config.beta1, 0.999))

        for ep in range(config.epoch):
            self.train()
            loss_epoch = []
            for idx in range(0, len(data_train_list), config.batch_size):
                batch_input = torch.tensor(
                    np.array([get_image(p, is_grayscale=False) for p in data_train_list[idx:idx + config.batch_size]]),
                    dtype=torch.float32).permute(0, 3, 1, 2).to(self.device)

                batch_wb_input = torch.tensor(
                    np.array([get_image(p, is_grayscale=False) for p in data_wb_train_list[idx:idx + config.batch_size]]),
                    dtype=torch.float32).permute(0, 3, 1, 2).to(self.device)

                batch_ce_input = torch.tensor(
                    np.array([get_image(p, is_grayscale=False) for p in data_ce_train_list[idx:idx + config.batch_size]]),
                    dtype=torch.float32).permute(0, 3, 1, 2).to(self.device)

                batch_gc_input = torch.tensor(
                    np.array([get_image(p, is_grayscale=False) for p in data_gc_train_list[idx:idx + config.batch_size]]),
                    dtype=torch.float32).permute(0, 3, 1, 2).to(self.device)

                batch_label = torch.tensor(
                    np.array([get_image(p, is_grayscale=False) for p in image_train_list[idx:idx + config.batch_size]]),
                    dtype=torch.float32).permute(0, 3, 1, 2).to(self.device)

                pred = self(batch_input, batch_wb_input, batch_ce_input, batch_gc_input)
                loss = self.compute_loss(pred, batch_label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_epoch.append(loss.item())

            print(f"Epoch {ep+1}, Loss: {np.mean(loss_epoch):.6f}")
            # Evaluate on 1 test image (optional: average more later)
            # Evaluation and image saving after each epoch
            self.eval()
            psnr_list = []
            ssim_list = []

            with torch.no_grad():
                for i in range(len(data_test_list)):
                    test_inputs = [
                        torch.tensor(np.array([get_image(p, is_grayscale=False)]), dtype=torch.float32)
                            .permute(0, 3, 1, 2).to(self.device)
                        for p in [data_test_list[i], data_wb_test_list[i], data_ce_test_list[i], data_gc_test_list[i]]
                    ]
                    label = torch.tensor(np.array([get_image(image_test_list[i], is_grayscale=False)]), dtype=torch.float32)\
                                .permute(0, 3, 1, 2).to(self.device)

                    pred = self(*test_inputs)

                    pred_np = (pred.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    label_np = (label.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

                    psnr = compare_psnr(label_np, pred_np, data_range=255)
                    ssim = compare_ssim(label_np, pred_np, win_size=5, channel_axis=-1, data_range=255)

                    psnr_list.append(psnr)
                    ssim_list.append(ssim)

                    # Save prediction (overwrite each epoch)
                    import imageio.v2 as imageio
                    save_path = os.path.join(self.sample_dir, f"test_{i:03d}.png")
                    imageio.imwrite(save_path, pred_np)

            # Print average evaluation scores
            avg_psnr = np.mean(psnr_list)
            avg_ssim = np.mean(ssim_list)
            print(f"        Average PSNR: {avg_psnr:.2f} dB, Average SSIM: {avg_ssim:.4f}")
            # Save model only if BOTH metrics improve
            if avg_psnr > best_psnr and avg_ssim > best_ssim:
                best_psnr = avg_psnr
                best_ssim = avg_ssim
                best_epoch = ep + 1
                torch.save(self.state_dict(), os.path.join(self.checkpoint_dir, "best_model.pth"))
                print(f"        ✅ Model saved at epoch {ep+1} (Best PSNR | SSIM)")
                print("\n📊 Best Evaluation Summary:")


        print(f"  ✅ Best Model at Epoch {best_epoch}")
        print(f"     - Average PSNR: {best_psnr:.2f} dB")
        print(f"     - Average SSIM: {best_ssim:.4f}")



