import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# importing libraries

import glob
import sys
import gc
from time import time
import cv2
from collections import Counter


import PIL as pil

import pandas as pd
import numpy as np
np.random.seed(42)

import matplotlib.pyplot as plt

import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from tqdm import tqdm
import wandb
wandb.login()

sys.path.insert(0,'../src/')

from data_process import *
from cvt import *
from model_train import *

label_diagram = pd.read_csv("../input/label_diagram.csv")

imgs_path = '../input/images_gz2/images/'
W, H, C = 224, 224, 4

file_list = glob.glob(os.path.join(imgs_path, '*.jpg'))
file_list = sorted(file_list)

# select files whose asset_id is the same as the one in the label_diagram
label_map = label_diagram.set_index('asset_id')["r1"].to_dict()

file_list = [(f,label_map.get(int(f.split('/')[-1].split('.')[0]), None)) for f in file_list if int(f.split('/')[-1].split('.')[0]) in label_diagram['asset_id'].values]

n = 50000/len(file_list)

images = [x[0] for x in file_list]
labels = [x[1] for x in file_list]

img_sub, im, labels_sub, l = train_test_split(images, labels, train_size=n, random_state=42, stratify=labels, shuffle=True)
print(len(img_sub), len(labels_sub))
print(img_sub[:5], labels_sub[:5])

counts = Counter(labels_sub)
print(counts)

class_mapping = {x : i for i, x in enumerate(sorted(label_diagram['r1'].unique()))}

if __name__ == '__main__':
    gmorph_d = galaxy_img_dataset(img_sub, label_diagram, label_mapping='r1', class_mapping=class_mapping)
    
    images = [x[0] for x in gmorph_d]
    labels = [x[1] for x in gmorph_d]

    x = np.array(images)
    y = np.array(labels)

    bs = 32
    # ---------
    print("preparing the data loaders")
    x_train, x_rem, y_train, y_rem = train_test_split(x,y, train_size=0.7, random_state=42, stratify=y, shuffle=True)

    x_valid, x_test, y_valid, y_test = train_test_split(x_rem,y_rem, test_size=0.34, random_state=42, stratify=y_rem, shuffle=True)

    x_train, y_train, x_valid, y_valid, x_test, y_test = map(torch.tensor, (x_train, y_train, x_valid, y_valid, x_test, y_test))

    train_ds = TensorDataset(x_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=16, pin_memory=True)

    valid_ds = TensorDataset(x_valid, y_valid)
    valid_dl = DataLoader(valid_ds, batch_size=bs, shuffle=True, num_workers=16,pin_memory=True)

    test_ds = TensorDataset(x_test, y_test)
    test_dl = DataLoader(test_ds, batch_size=bs, shuffle=True, num_workers=16,pin_memory=True)

    epochs = 50
    lr = 0.0001
    tmax = epochs
    device= 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 32
    embed_size = 64

    gmorph_model = CvT(embed_size, len(class_mapping))
    optimizer = torch.optim.AdamW(gmorph_model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, tmax, eta_min=1e-6)
    loss_func = nn.CrossEntropyLoss()

    results, results_class, train_pred, train_true, train_probs, valid_pred, valid_true, valid_probs = train_model(epochs, gmorph_model, train_dl, valid_dl, loss_func, optimizer, scheduler, device, save_name=f'test_final')

    del gmorph_model
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()