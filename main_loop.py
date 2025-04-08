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
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

wandb.login()

sys.path.insert(0,'../src/')

import pytorch_lightning as pl

from data_processing import *
from model_train import *
from labeling_system import *
import cvt as cvt
import cvt_benchmark as cvtb
# ----------------

reference_images = pd.read_csv('../input/filename_mapping.csv')

main_catalogue = pd.read_csv('../input/gz2_classes.csv')

main_catalogue.drop(main_catalogue[main_catalogue['gz2class'] == 'A'].index, inplace=True)

main_catalogue = main_catalogue.merge(
    reference_images[['objid', 'asset_id']], 
    left_on='dr7objid', 
    right_on='objid', 
    how='left'
).drop(columns=['objid'])  # Drop extra 'objid' column after merging
main_catalogue = main_catalogue.sort_values(by=['asset_id']).reset_index(drop=True)

# ===========

print('loaded the catalogues')
soft_label_dict_run1 = create_label_dict1(main_catalogue)
soft_label_dict_run2 = create_label_dict2(main_catalogue)
print(soft_label_dict_run1[list(soft_label_dict_run1.keys())[0]])
print(soft_label_dict_run2[list(soft_label_dict_run2.keys())[0]])

print('soft labels created')
hard_run1 = create_hard_labels(soft_label_dict_run1)
hard_run2 = create_hard_labels(soft_label_dict_run2)

print(hard_run1[list(hard_run1.keys())[0]])
print(hard_run2[list(hard_run2.keys())[0]])
print('hard labels created')
soft_run1_conf, soft_run1_spur = section_spurious(main_catalogue, soft_label_dict_run1, entropy_threshold=1.5)
print('spurios stuff detected')
runs = [
    {"soft_labels": soft_label_dict_run1, "hard_labels": hard_run1, "num_classes": 4},
    {"soft_labels": soft_label_dict_run2, "hard_labels": hard_run2, "num_classes": 6},
]
# ===========

imgs_path = '../input/images_gz2/images/'
W, H, C = 224, 224, 4

file_list = create_file_list(imgs_path, soft_run1_conf)
print('file list created')
n = 10000

img_sub, labels_sub = data_setup(file_list, hard_run1, n)
traino, valido, testo = split_data(img_sub, labels_sub)
print('data setup complete!')
# ==================

# LOOP
# params
epochs = 40
lr = 1e-4
tmax = epochs
device= 'cuda' if torch.cuda.is_available() else 'cpu'
bs = 32
embed_size = 64
use_soft_labels = True
m = 3

if __name__ == '__main__':
    print('Loop started')
    
    print('----------------')
    #  INITIAL DATA SETUP
    # =====================

    train_dl, valid_dl, test_dl, y_train, y_valid, y_test = create_data_loaders(traino, valido, testo, hard_run1, soft_label_dict_run1, bs, aux_train=None, aux_valid=None, aux_test=None)
    
    # SAVING SETUP
    # ======================

    results_runs = []
    results_runs_class = []

    outputs, labels = [], []

    axt = torch.zeros(len(y_train))
    axv = torch.zeros(len(y_valid))
    axte = torch.zeros(len(y_test))

    # ======================
    for i in range(m-1):
        print(f'ITERATION{i}')
        config = runs[i]
        
        gmorph_model = cvt.CvT(embed_size, config['num_classes'], hint=(i>0))

        optimizer = torch.optim.NAdam(gmorph_model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, tmax, eta_min=1e-6)
        loss_func1 = nn.KLDivLoss(reduction='batchmean')
        loss_func2 = nn.CrossEntropyLoss()

        results, results_class, train_pred, train_true, train_probs, valid_pred, valid_true, valid_probs = train_model(epochs, gmorph_model, train_dl, valid_dl, loss_func1, loss_func2, optimizer, scheduler, device, save_name=f'r{i+1}FINAL_Soft')

        results_runs.append((f'r{i+1}', results))
        results_runs_class.append((f'r{i+1}', results_class))

        y_true, preds = test_model(test_dl, gmorph_model, device)
        outputs.append(preds.cpu().detach())
        labels.append(y_true.cpu().detach())

        del gmorph_model
        gc.collect()
        torch.cuda.empty_cache()

        # PREP FOR NEXT ROUND
        if i+2 < m:
            print('Updating data')
            print('---------------')

            axt += y_train.argmax(dim=1)
            axv += y_valid
            axte += torch.tensor(preds.detach().cpu().numpy())

            train_dl, valid_dl, test_dl, y_train, y_valid, y_test = create_data_loaders(traino, valido, testo, runs[i+1]["hard_labels"], runs[i+1]["soft_labels"], bs, aux_train=axt, aux_valid=axv, aux_test=axte)

    outputs = np.array([x.cpu().detach().numpy() for x in outputs])
    labels = np.array([x.cpu().detach().numpy() for x in labels])

    np.save('../output/results_runs_final_final.npy', results_runs, allow_pickle=True)
    np.save('../output/results_runs_class_final_final.npy', results_runs_class, allow_pickle=True)
    np.save('../output/outputs_test_final_final.npy', outputs, allow_pickle=True)
    np.save('../output/labels_test_final_final.npy', labels, allow_pickle=True)
    
    print('Loop is finished!')

    # =============
    # BENCHMARK
    # =============
    '''
    benchmark_soft_labels = {
    int(row["asset_id"]): create_benchmark_soft_labels(row)
    for _, row in main_catalogue.iterrows()
    }

    benchmark_hard_labels = {
        asset_id: int(np.argmax(label)) 
        for asset_id, label in benchmark_soft_labels.items()
    }

    train_dl, valid_dl, test_dl = create_data_loaders_bench(traino, valido, testo, benchmark_hard_labels, bs, soft_labels_dict=benchmark_soft_labels)

    gmorph_model = cvtb.CvT_bench(embed_size, 8)
    optimizer = torch.optim.NAdam(gmorph_model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, tmax, eta_min=1e-5)
    loss_func1 = nn.KLDivLoss(reduction='batchmean')
    loss_func2 = nn.CrossEntropyLoss()

    results, results_class, train_pred, train_true, train_probs, valid_pred, valid_true, valid_probs = train_model(epochs, gmorph_model, train_dl, valid_dl, loss_func1, loss_func2, optimizer, scheduler, device, save_name='benchmark_final')

    y_true, preds = test_model(test_dl, gmorph_model, device)

    del gmorph_model
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()

    outputs = np.array([x.cpu().detach().numpy() for x in preds])
    labels = np.array([x.cpu().detach().numpy() for x in y_true])

    np.save('../output/results_runs_bench_final_final.npy', results_runs, allow_pickle=True)
    np.save('../output/results_runs_class_bench_final_final.npy', results_runs_class, allow_pickle=True)
    np.save('../output/outputs_test_bench_final_final.npy', outputs, allow_pickle=True)
    np.save('../output/labels_test_bench_final_final.npy', labels, allow_pickle=True)

    print('Completely done!')

    '''