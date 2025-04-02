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

# loading the label_diagram chart for label assignment
label_diagram = pd.read_csv("../input/label_diagram.csv")

imgs_path = '../input/images_gz2/images/'
W, H, C = 224, 224, 4

file_list = create_file_list(imgs_path, label_diagram)

# assortment of label_maps
runs = {f: () for f in file_list}

for i in label_diagram.columns:
    if i == 'asset_id':
        continue
    # Create a mapping from asset_id to the current label value
    label_map = label_diagram.set_index('asset_id')[i].to_dict()
    # For each file, append the label value to the tuple already stored (or create a new tuple)
    for f in file_list:
        asset_id = int(f.split('/')[-1].split('.')[0]) # select the asset_id
        label_val = label_map.get(asset_id, None) # get the label value
        runs[f] = runs.get(f, ()) + (label_val,) # connect the filename and the label value
        
print('Labels assigned')
print(runs[file_list[0]])

n = 50000/len(file_list)

# ==================

# LOOP
# params
epochs = 60
lr = 0.0001
tmax = 20
device= 'cuda' if torch.cuda.is_available() else 'cpu'
bs = 32
embed_size = 64
m = 3

if __name__ == '__main__':
    print('Loop started')

    print('----------------')
    #  INITIAL DATA SETUP
    # =====================
    class_mapping, labels_orig, train_dl, valid_dl, test_dl = data_setup_run(runs, file_list, label_diagram, 0, n, bs)

    # SAVING SETUP
    # ======================

    results_runs = []
    results_runs_class = []


    # ======================
    for i in range(1, m):
        print(f'ITERATION{i}')

        # MODEL
        if (i > 1):
            gmorph_model = CvT(embed_size, len(class_mapping), hint=True)
        else:
            gmorph_model = CvT(embed_size, len(class_mapping))

        optimizer = torch.optim.AdamW(gmorph_model.parameters(), lr=lr, weight_decay=0.07)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, tmax, eta_min=1e-6)
        loss_func = nn.CrossEntropyLoss()

        results, results_class, train_pred, train_true, train_probs, valid_pred, valid_true, valid_probs = train_model(epochs, gmorph_model, train_dl, valid_dl, loss_func, optimizer, scheduler, device, save_name=f'r{i}_loop_final')

        results_runs.append(results)
        results_runs_class.append(results_class)

        del gmorph_model
        del optimizer
        gc.collect()
        torch.cuda.empty_cache()

        # PREP FOR NEXT ROUND
        print('Updating data')
        print('---------------')
        labels_orig_num = [class_mapping[x] for x in labels_orig]
        print(labels_orig_num[:2])
        data_setup_run(runs, file_list, label_diagram, i, n, bs, labels_orig_num)

