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
import random

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

from data_processing import *
from model_train import *
import cvt as cvt
import cvt_benchmark as cvtb
# ----------------

# loading the label_diagram chart for label assignment
label_diagram = pd.read_csv("../input/label_diagram.csv")

imgs_path = '../input/images_gz2/images/'
W, H, C = 224, 224, 4

file_list = create_file_list(imgs_path, label_diagram)

n = 10000

img_sub, labels_sub = data_setup(file_list, label_diagram, n)
traino, valido, testo = split_data(img_sub, labels_sub)

# ==================

# LOOP
# params
epochs = 60
lr = 0.0001
tmax = epochs
device= 'cuda' if torch.cuda.is_available() else 'cpu'
bs = 32
embed_size = 64
m = 3

if __name__ == '__main__':
    print('Loop started')

    print('----------------')
    #  INITIAL DATA SETUP
    # =====================
    
    train_dl, valid_dl, test_dl, y_train, y_valid, y_test, class_mapping = create_data_loaders(traino, valido, testo, label_diagram, 0, bs, aux_train=None, aux_valid=None, aux_test=None)
    
    # SAVING SETUP
    # ======================

    results_runs = []
    results_runs_class = []

    outputs, labels = [], []

    axt = torch.zeros(len(y_train))
    axv = torch.zeros(len(y_valid))
    axte = torch.zeros(len(y_test))

    # ======================
    for i in range(1, m):
        print(f'ITERATION{i}')

        # MODEL
        if (i > 1):
            gmorph_model = cvt.CvT(embed_size, len(class_mapping), hint=True)
        else:
            gmorph_model = cvt.CvT(embed_size, len(class_mapping))

        optimizer = torch.optim.NAdam(gmorph_model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, tmax, eta_min=1e-5)
        loss_func = nn.CrossEntropyLoss()

        results, results_class, train_pred, train_true, train_probs, valid_pred, valid_true, valid_probs = train_model(epochs, gmorph_model, train_dl, valid_dl, loss_func, optimizer, scheduler, device, save_name=f'r{i}_FINAL_balanced')

        results_runs.append((f'r{i}', results))
        results_runs_class.append(results_class)

        y_true, preds = test_model(test_dl, gmorph_model, device)
        outputs.append(preds.cpu().detach())
        labels.append(y_true.cpu().detach())

        del gmorph_model
        del optimizer
        gc.collect()
        torch.cuda.empty_cache()

        # PREP FOR NEXT ROUND
        print('Updating data')
        print('---------------')

        axt += y_train
        axv += y_valid
        axte += torch.tensor(preds.detach().cpu().numpy())

        train_dl, valid_dl, test_dl, y_train, y_valid, y_test, class_mapping = create_data_loaders(traino, valido, testo, label_diagram, i, bs, aux_train=axt, aux_valid=axv, aux_test=axte)
    
    print('Loop is finished!')

    results_runs = np.array(results_runs)
    results_runs_class = np.array(results_runs_class)

    outputs = np.array([x.cpu().detach().numpy() for x in outputs])
    labels = np.array([x.cpu().detach().numpy() for x in labels])

    np.save('../output/results_runs.npy', results_runs, allow_pickle=True)
    np.save('../output/results_runs_class.npy', results_runs_class, allow_pickle=True)
    np.save('../output/outputs_test.npy', outputs, allow_pickle=True)
    np.save('../output/labels_test.npy', labels, allow_pickle=True)

    # =============
    # BENCHMARK
    # =============

    labels_bench = [(label_diagram["r1"][x]+label_diagram["r2"][x]) for x in range(label_diagram.shape[0])]
    label_mapping = {label_diagram['asset_id'][x]: labels_bench[x] for x in range(len(labels_bench))}

    train_dl, valid_dl, test_dl, class_mapping = create_data_loaders_bench(traino, valido, testo, labels_bench, label_mapping, bs)

    gmorph_model = cvtb.CvT_bench(embed_size, len(class_mapping))
    optimizer = torch.optim.NAdam(gmorph_model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, tmax, eta_min=1e-5)
    loss_func = nn.CrossEntropyLoss()

    results, results_class, train_pred, train_true, train_probs, valid_pred, valid_true, valid_probs = train_model(epochs, gmorph_model, train_dl, valid_dl, loss_func, optimizer, scheduler, device, save_name='benchmark_final')

    y_true, preds = test_model(test_dl, gmorph_model, device)

    del gmorph_model
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()

    outputs = np.array([x.cpu().detach().numpy() for x in preds])
    labels = np.array([x.cpu().detach().numpy() for x in y_true])

    np.save('../output/results_runs_bench.npy', results_runs, allow_pickle=True)
    np.save('../output/results_runs_class_bench.npy', results_runs_class, allow_pickle=True)
    np.save('../output/outputs_test_bench.npy', outputs, allow_pickle=True)
    np.save('../output/labels_test_bench.npy', labels, allow_pickle=True)

    print('Completely done!')