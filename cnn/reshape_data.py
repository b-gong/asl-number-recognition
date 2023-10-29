from PIL import Image
import numpy as np
import os

import copy
import sys
import time
from tqdm import tqdm

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from collections import defaultdict

torch.use_deterministic_algorithms(True)
torch.manual_seed(0)
file = 'asl_data.npy'
X_data = defaultdict(list)
y_data = defaultdict(list)
all_data = {}

def load_and_preprocess_images(path, size=(64, 64)):
    """Load and preprocess images from a given path."""
    images = []
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        img = Image.open(img_path)
        images.append(np.array(img))
    return images

for number in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
    for subset in ['train', 'dev', 'test']:
        images = load_and_preprocess_images(os.path.join('../dataset', subset, number))
        for image in images:
            X_data[subset].append(image)
            y_data[subset].append(int(number))

all_data['X_train'] = np.array(X_data['train'])
all_data['y_train'] = np.array(y_data['train'])
all_data['X_dev'] = np.array(X_data['dev'])
all_data['y_dev'] = np.array(y_data['dev'])
all_data['X_test'] = np.array(X_data['test'])
all_data['y_test'] = np.array(y_data['test'])

N, _, _ = all_data['X_train'].shape
all_data['X_train'] = all_data['X_train'].reshape(N, -1)
N, _, _ = all_data['X_dev'].shape
all_data['X_dev'] = all_data['X_dev'].reshape(N, -1)
N, _, _ = all_data['X_test'].shape
all_data['X_test'] = all_data['X_test'].reshape(N, -1)

np.save(file, all_data)

print(f"Data saved in {file}")
