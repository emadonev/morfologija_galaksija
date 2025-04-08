import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import sys
import gc
from time import time
from collections import Counter
import pandas as pd
import numpy as np
np.random.seed(42)

import matplotlib.pyplot as plt

import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from tqdm import tqdm
import wandb
import pickle


wandb.login()

sys.path.insert(0,'../src/')

# -----
import datap_efficient_runs as drun
import model_train_runs as trrun
from labeling_system import *
import cvt_OHE_full_att as cvt
# -----

main_runs = pd.read_csv('../input/main_runs.csv')

# ------

soft_label_dict_run1 = create_label_dict1(main_runs)
soft_label_dict_run2 = create_label_dict2(main_runs)

soft_run1_conf, soft_run1_spur = section_spurious(soft_label_dict_run1, num=3)
soft_run2_conf, soft_run2_spur = section_spurious(soft_label_dict_run2, num=7)

hard_run1_conf = create_hard_labels(soft_run1_conf)
hard_run2_conf = create_hard_labels(soft_run2_conf)

print(Counter(list(hard_run1_conf.values())))
print(Counter(list(hard_run2_conf.values())))

imgs_path = '../input/images_gz2/images/'
W, H, C = 224, 224, 4

conf_file_list = create_file_list(imgs_path, soft_run1_conf, soft_run2_conf)

# ------

n = 5000
bs = 128
images_orig, labels_orig = drun.data_setup(conf_file_list, hard_run2_conf, n)
traino, valido, testo, y_traino, y_valido, y_testo = drun.split_data(images_orig, labels_orig)

print(Counter(y_traino))

# -----
epochs = 50
lr = 1e-4
tmax = epochs
device= 'cuda' if torch.cuda.is_available() else 'cpu'
embed_size = 64

#RUN 1
# ==========
train_dl_run1, valid_dl_run1, test_dl_run1, train_coarse1, valid_coarse1, test_coarse1 = drun.create_dali_iterators(traino, valido, testo, hard_run1_conf, bs)

gmorph_model = cvt.CvT_cyclic(embed_size, 3, hint=False)

optimizer = torch.optim.AdamW(gmorph_model.parameters(), lr=lr, weight_decay=0.04, betas=(0.9, 0.999), eps=1e-8)
warmup_epochs = 5
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=lr,
    epochs=epochs,
    steps_per_epoch=len(train_dl_run1),
    pct_start=warmup_epochs/epochs,
    anneal_strategy='cos',
    div_factor=25.0,
    final_div_factor=10000.0
)
max_grad_norm = 1.0
loss_func = nn.CrossEntropyLoss()

results, results_class, train_pred, train_true, train_probs, train_galaxy_ids, valid_pred, valid_true, valid_probs, valid_galaxy_ids = trrun.train_model(epochs, gmorph_model, train_dl_run1, 
valid_dl_run1, loss_func, optimizer, scheduler, device, max_grad_norm, 
save_name='RUN_01_full_final', train_coarse=train_coarse1, valid_coarse=valid_coarse1, 
num_classes=3, model_path='../output/run1/')

with open('../output/run1/results_runs_bench_final_true.pkl', 'wb') as f:
    pickle.dump(results, f)

with open('../output/run1/results_runs_class_bench_final_true.pkl', 'wb') as f:
    pickle.dump(results_class, f)

# UPDATING THE DATA
# ------

del train_dl_run1, valid_dl_run1, test_dl_run1
del gmorph_model, optimizer, scheduler
torch.cuda.empty_cache()
gc.collect()

train_dl_run2, valid_dl_run2, test_dl_run2, train_coarse2, valid_coarse2, test_coarse2 = drun.create_dali_iterators(traino, valido, testo, hard_run2_conf, bs, previous=train_coarse1)

gmorph_model = cvt.CvT_cyclic(embed_size, 7, hint=True)

optimizer = torch.optim.AdamW(gmorph_model.parameters(), lr=lr, weight_decay=0.04, betas=(0.9, 0.999), eps=1e-8)
warmup_epochs = 5
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=lr,
    epochs=epochs,
    steps_per_epoch=len(train_dl_run2),
    pct_start=warmup_epochs/epochs,
    anneal_strategy='cos',
    div_factor=25.0,
    final_div_factor=10000.0
)
max_grad_norm = 1.0
loss_func = nn.CrossEntropyLoss()

results, results_class, train_pred, train_true, train_probs, train_galaxy_ids, valid_pred, valid_true, valid_probs, valid_galaxy_ids = trrun.train_model(epochs, gmorph_model, train_dl_run2, 
valid_dl_run2, loss_func, optimizer, scheduler, device, max_grad_norm, 
save_name='RUN_02_full_final', train_coarse=train_coarse2, 
valid_coarse=valid_coarse2, num_classes=7, model_path='../output/run2/')

with open('../output/run2/results_runs_bench_final_true.pkl', 'wb') as f:
    pickle.dump(results, f)

with open('../output/run2/results_runs_class_bench_final_true.pkl', 'wb') as f:
    pickle.dump(results_class, f)