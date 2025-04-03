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

from data_processing import *
from cvt import *
from model_train import *
# ----------------

# loading the label_diagram chart for label assignment
label_diagram = pd.read_csv("../input/label_diagram.csv")

imgs_path = '../input/images_gz2/images/'
W, H, C = 224, 224, 4

file_list = create_file_list(imgs_path, label_diagram)

n = 40000/len(file_list)

img_sub, labels_sub = data_setup(file_list, label_diagram, n)
traino, valido, testo = split_data(img_sub, labels_sub)

# ==================

# LOOP
# params
epochs = 50
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
            gmorph_model = CvT(embed_size, len(class_mapping), hint=True)
        else:
            gmorph_model = CvT(embed_size, len(class_mapping))

        optimizer = torch.optim.NAdam(gmorph_model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, tmax, eta_min=1e-6)
        loss_func = nn.CrossEntropyLoss()

        results, results_class, train_pred, train_true, train_probs, valid_pred, valid_true, valid_probs = train_model(epochs, gmorph_model, train_dl, valid_dl, loss_func, optimizer, scheduler, device, save_name=f'r{i}_FINAL')

        results_runs.append((f'r{i}', results))
        results_runs_class.append(results_class)

        out, lab = test_model(test_dl, gmorph_model, device)
        outputs.append(out.cpu().detach())
        labels.append(lab.cpu().detach())

        del gmorph_model
        del optimizer
        gc.collect()
        torch.cuda.empty_cache()

        # PREP FOR NEXT ROUND
        print('Updating data')
        print('---------------')

        axt += y_train
        axv += y_valid
        axte += y_test

        train_dl, valid_dl, test_dl, y_train, y_valid, y_test, class_mapping = create_data_loaders(traino, valido, testo, label_diagram, i, bs, aux_train=axt, aux_valid=axv, aux_test=axte)
    
    print('Loop is finished!')

    results_runs = np.array(results_runs)
    results_runs_class = np.array(results_runs_class)

    print(outputs[0], labels[0])
    print(outputs[1], labels[1])

    outputs = np.array([x.cpu().detach().numpy() for x in outputs])
    labels = np.array([x.cpu().detach().numpy() for x in labels])

    np.save('../output/results_runs.npy', results_runs, allow_pickle=True)
    np.save('../output/results_runs_class.npy', results_runs_class, allow_pickle=True)
    np.save('../output/outputs_test.npy', outputs, allow_pickle=True)
    np.save('../output/labels_test.npy', labels, allow_pickle=True)