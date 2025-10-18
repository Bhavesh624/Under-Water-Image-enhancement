import os
import glob
import h5py
import random
import cv2 as cv
import numpy as np
from PIL import Image
import imageio
import torch

def transform(images):
    return np.array(images) / 127.5 - 1.

def inverse_transform(images):
    return (images + 1.) / 2

def prepare_data(dataset):
    filenames = os.listdir(dataset)
    data_dir = os.path.join(os.getcwd(), dataset)
    data = glob.glob(os.path.join(data_dir, "*.jpg"))
    data += glob.glob(os.path.join(data_dir, "*.png"))
    return data

def imread(path, is_grayscale=False):
    if is_grayscale:
        return imageio.imread(path, pilmode='L').astype(np.float32)
    else:
        return imageio.imread(path).astype(np.float32)

def imsave(image, path):
    imsaved = (inverse_transform(image)).astype(np.float32)
    return imageio.imwrite(path, imsaved)

def get_image(image_path, is_grayscale):
    if is_grayscale:
        image = Image.open(image_path).convert('L')
    else:
        image = Image.open(image_path).convert('RGB')
    return np.array(image).astype(np.float32) / 255.0

def get_label(image_path, is_grayscale):
    return get_image(image_path, is_grayscale)

def imsave_label(image, path):
    return imageio.imwrite(path, (image * 255).astype(np.uint8))

def white_balance(img, percent=0):
    out_channels = []
    cumstops = (
        img.shape[0] * img.shape[1] * percent / 200.0,
        img.shape[0] * img.shape[1] * (1 - percent / 200.0)
    )
    for channel in cv.split(img):
        cumhist = np.cumsum(cv.calcHist([channel], [0], None, [256], (0, 256)))
        low_cut, high_cut = np.searchsorted(cumhist, cumstops)
        lut = np.concatenate((
            np.zeros(low_cut),
            np.around(np.linspace(0, 255, high_cut - low_cut + 1)),
            255 * np.ones(255 - high_cut)
        ))
        out_channels.append(cv.LUT(channel, lut.astype('uint8')))
    img = cv.merge(out_channels)
    return img

def adjust_gamma(image, gamma=0.7):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv.LUT(image.astype(np.uint8), table)