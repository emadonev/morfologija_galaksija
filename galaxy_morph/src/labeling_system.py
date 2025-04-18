import glob
import pandas as pd
import numpy as np
import os
import PIL as pil
import concurrent.futures
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import train_test_split
from torch.utils.data.distributed import DistributedSampler
import torch
import matplotlib.pyplot as plt
import cv2
import random
from scipy.stats import entropy

# ==============

W, H = 224, 224

# =============

# SOFT LABELING

# =============

# RUN 1
# =======
# E, S, Se
# --------------

def run1_soft_labels(row):
    # E 
    p_e = row["a01_smooth"]

    # S
    p_s = (row["a02_features_disk"] * row["a04_edgeon_no"] * row['a08_spiral'])

    # Se
    p_se = (row["a02_features_disk"] * row['a04_edgeon_yes'])

    # Normalize
    total = p_e + p_s + p_se
    if total == 0:
        return np.array([1.0, 0.0, 0.0])  # fallback: assume elliptical

    return np.array([p_e, p_s, p_se]) / total

# RUN 2
# ========

def run2_soft_labels(row):

    pr = (row["a01_smooth"] * row['a16_completely_round'])

    pi = (row["a01_smooth"] * row['a17_in_between'])

    pc = (row["a01_smooth"] * row['a18_cigar_shaped'])

    # -----

    pBar = (row["a02_features_disk"] * row["a04_edgeon_no"] * row['a08_spiral'] * row['a06_bar'])

    pnoBar = (row["a02_features_disk"] * row["a04_edgeon_no"] * row['a08_spiral'] * row['a07_no_bar'])

    # -----

    pSeBulge = (row["a02_features_disk"] * row['a04_edgeon_yes'] * row['a25_round_bulge'] + row["a02_features_disk"] * row['a04_edgeon_yes'] * row["a26_boxy_bulge"])

    pSenoB = (row["a02_features_disk"] * row['a04_edgeon_yes'] * row["a27_no_bulge"])

    # Normalize
    total = pr + pi + pc + pBar + pnoBar + pSeBulge + pSenoB
    if total == 0:
        return np.array([0.0,0.0,0.0,0.0,0.0])

    return np.array([pr, pi, pc, pBar, pnoBar, pSeBulge, pSenoB]) / total

# ----
def get_label_entropy(soft_label, num):
    max_entropy = np.log2(num)
    return (entropy(soft_label, base=2)) / max_entropy if max_entropy > 0 else 0

def create_label_dict1(data):
    soft_label_dict = {
    int(row["asset_id"]): run1_soft_labels(row)
    for _, row in data.iterrows()
    }
    return soft_label_dict

def create_label_dict2(data):
    soft_label_dict = {
    int(row["asset_id"]): run2_soft_labels(row)
    for _, row in data.iterrows()
    }
    return soft_label_dict

def section_spurious(soft_label_dict, num, entropy_threshold=0.7):
    confident = {}
    spurious = {}

    for asset_id, label in soft_label_dict.items():
        if get_label_entropy(label, num) > entropy_threshold:
            spurious[asset_id] = label
        else:
            confident[asset_id] = label

    return confident, spurious

def create_file_list(imgs_path, label_dict1, label_dict2):
    file_list = glob.glob(os.path.join(imgs_path, '*.jpg'))
    file_list = sorted(file_list)

    file_list = [(f, int(f.split('/')[-1].split('.')[0])) for f in file_list if (int(f.split('/')[-1].split('.')[0]) in label_dict1) and (int(f.split('/')[-1].split('.')[0]) in label_dict2)]

    return file_list

# =============

# HARD LABELING

# =============

# we take the previous labels and just make them into hard predictions with argmax

def create_hard_labels(labels_dict):
    hard_labels = {}
    for asset_id, label in labels_dict.items():
        hard_labels[asset_id] = int(np.argmax(label))
    return hard_labels

# =============

# BENCHMARK

# =============

benchmark_classes = ["S", "SB", "Ei", "Er", "Ec", "Seb", "Sen", "Sei"]
benchmark_class_map = {name: i for i, name in enumerate(benchmark_classes)}

def create_benchmark_soft_labels(row):
    probs = np.zeros(8, dtype=np.float32)

    # Smooth-related
    p_smooth = row["t01_smooth_or_features_a01_smooth_debiased"]
    p_edgeon = row["t02_edgeon_a04_yes_debiased"]

    # Roundedness of smooth galaxies
    p_er = p_smooth * row["t07_rounded_a16_completely_round_debiased"]
    p_ei = p_smooth * row["t07_rounded_a17_in_between_debiased"]
    p_ec = p_smooth * row["t07_rounded_a18_cigar_shaped_debiased"]

    # Edge-on + bulge shape
    p_seb = p_edgeon * row["t09_bulge_shape_a26_boxy_debiased"]
    p_sen = p_edgeon * row["t09_bulge_shape_a27_no_bulge_debiased"]
    p_sei = p_edgeon * row["t09_bulge_shape_a25_rounded_debiased"]

    # Disk + spiral structure
    p_features = row["t01_smooth_or_features_a02_features_or_disk_debiased"]
    p_spiral = row["t04_spiral_a08_spiral_debiased"]

    p_s = p_features * p_spiral * row["t03_bar_a07_no_bar_debiased"]
    p_sb = p_features * p_spiral * row["t03_bar_a06_bar_debiased"]

    # Assign
    probs[benchmark_class_map["S"]] = p_s
    probs[benchmark_class_map["SB"]] = p_sb
    probs[benchmark_class_map["Er"]] = p_er
    probs[benchmark_class_map["Ei"]] = p_ei
    probs[benchmark_class_map["Ec"]] = p_ec
    probs[benchmark_class_map["Seb"]] = p_seb
    probs[benchmark_class_map["Sen"]] = p_sen
    probs[benchmark_class_map["Sei"]] = p_sei

    total = probs.sum()
    if total == 0:
        probs[benchmark_class_map["S"]] = 1.0  # fallback
    else:
        probs /= total

    return probs


