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
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from tqdm import tqdm
import wandb


wandb.login()

sys.path.insert(0,'../src/')

# -----
from data_processing import *
from model_train import *
from labeling_system import *
import cvt as cvt
import cvt_benchmark as cvtb
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

n = 5000
bs = 32
images_orig, labels_orig = data_setup(conf_file_list, hard_run2_conf, n)
traino, valido, testo, y_traino, y_valido, y_testo = split_data(images_orig, labels_orig)

print(Counter(y_traino))

train_dl, valid_dl, test_dl, y_train, y_valid, y_test = create_data_loaders(traino, valido, testo, hard_run1_conf, soft_label_dict_run1, bs, aux_train=None, aux_valid=None, aux_test=None)

# -----

epochs = 25
epochs2 = 30
lr = 1e-4
tmax = epochs
tmax2 = epochs2
device= 'cuda' if torch.cuda.is_available() else 'cpu'
bs = 32
embed_size = 64
use_soft_labels = True

results_runs = []
results_runs_class = []

outputs, labels = [], []

axt = torch.zeros(len(y_train))
axv = torch.zeros(len(y_valid))
axte = torch.zeros(len(y_test))

gmorph_model = cvt.CvT(embed_size, 3, hint=False)

optimizer = torch.optim.NAdam(gmorph_model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, tmax, eta_min=1e-6)
loss_func1 = nn.KLDivLoss(reduction='batchmean')
loss_func2 = nn.CrossEntropyLoss()

results, results_class, train_pred, train_true, train_probs, valid_pred, valid_true, valid_probs = train_model(epochs, gmorph_model, train_dl, valid_dl, loss_func1, loss_func2, optimizer, scheduler, device, save_name='Run_01_final')

results_runs.append(('r1', results))
results_runs_class.append(('r1', results_class))

y_true, preds = test_model(test_dl, gmorph_model, device)
outputs.append(preds.cpu().detach())
labels.append(y_true.cpu().detach())

del gmorph_model
gc.collect()
torch.cuda.empty_cache()

outputs = np.array([x.cpu().detach().numpy() for x in outputs])
labels = np.array([x.cpu().detach().numpy() for x in labels])

np.save('../output/results_runs_run1_final.npy', results_runs, allow_pickle=True)
np.save('../output/results_runs_class_run1_final.npy', results_runs_class, allow_pickle=True)
np.save('../output/outputs_test_run1_final.npy', outputs, allow_pickle=True)
np.save('../output/labels_test_run1_final.npy', labels, allow_pickle=True)


# RUN 2
axt += y_train.argmax(dim=1)
axv += y_valid
axte += torch.tensor(preds.detach().cpu().numpy())

train_dl, valid_dl, test_dl, y_train, y_valid, y_test = create_data_loaders(traino, valido, testo, hard_run2_conf, soft_label_dict_run2, bs, aux_train=axt, aux_valid=axv, aux_test=axte)

results_runs = []
results_runs_class = []

outputs, labels = [], []

gmorph_model = cvt.CvT(embed_size, 7, hint=True)

optimizer = torch.optim.NAdam(gmorph_model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, tmax2, eta_min=1e-6)
loss_func1 = nn.KLDivLoss(reduction='batchmean')
loss_func2 = nn.CrossEntropyLoss()

results, results_class, train_pred, train_true, train_probs, valid_pred, valid_true, valid_probs = train_model(epochs2, gmorph_model, train_dl, valid_dl, loss_func1, loss_func2, optimizer, scheduler, device, save_name='Run_02_final')

results_runs.append(('r2', results))
results_runs_class.append(('r2', results_class))

y_true, preds = test_model(test_dl, gmorph_model, device)
outputs.append(preds.cpu().detach())
labels.append(y_true.cpu().detach())

del gmorph_model
gc.collect()
torch.cuda.empty_cache()

outputs = np.array([x.cpu().detach().numpy() for x in outputs])
labels = np.array([x.cpu().detach().numpy() for x in labels])

np.save('../output/results_runs_run2_final.npy', results_runs, allow_pickle=True)
np.save('../output/results_runs_class_run2_final.npy', results_runs_class, allow_pickle=True)
np.save('../output/outputs_test_run2_final.npy', outputs, allow_pickle=True)
np.save('../output/labels_test_run2_final.npy', labels, allow_pickle=True)
