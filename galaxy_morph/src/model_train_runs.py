# importing libraries

from typing import Any


import os
import glob
import sys
sys.path.insert(0,'../src/')
import gc
from time import time

import wandb

import pandas as pd
import numpy as np

np.random.seed(42)

import matplotlib.pyplot as plt

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch.nn as nn
import torch.nn.functional as F

# ---------------------


def accuracy(predictions, labels, treshold)->int:
    preds_b = (predictions > treshold).float()
    return (preds_b == labels).sum().item()

def to_one_hot(labels, num_classes, device='cuda'):
    # Ensure labels are on the correct device
    labels = labels.to(device)
    # Create one-hot tensor on the same device
    one_hot = torch.zeros(labels.size(0), num_classes, device=device)
    one_hot.scatter_(1, labels.view(-1, 1), 1)
    return one_hot

def train_epoch(model, optimizer, data_loader, loss_func, device, max_grad_norm, coarse_labels=None, num_classes=7):
    total_loss = 0
    total_samples = 0
    y_true = []
    y_pred = []
    y_probs = []
    galaxy_ids = []

    model.train()
    for i, data in enumerate(data_loader):
        imgs = data[0]['data'].to(device)
        labels = data[0]['label'].to(device).view(-1).long()
        batch_galaxy_ids = data[0]['galaxy_id'].cpu().numpy()
        batch_size = labels.size(0)
        total_samples += batch_size

        optimizer.zero_grad()
        
        with torch.autocast(device_type=device, dtype=torch.float16):
            if coarse_labels is not None:
                # Convert coarse labels to one-hot encoding
                batch_coarse = torch.tensor(coarse_labels[i*data_loader.batch_size:(i+1)*data_loader.batch_size])
                batch_coarse = to_one_hot(batch_coarse, num_classes, device=device)
                outputs = model(imgs, coarse_label=batch_coarse)
            else:
                outputs = model(imgs)
            
            # Extract logits from CvTOutput
            logits = outputs.logits
            loss = loss_func(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        
        total_loss += loss.item() * batch_size

        probabilities = F.softmax(logits, dim=1)
        preds = probabilities.argmax(dim=1)
        
        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_pred.extend(preds.detach().cpu().numpy().tolist())
        y_probs.extend(probabilities.detach().cpu().numpy().tolist())
        galaxy_ids.extend(batch_galaxy_ids.tolist())

        loss_train = total_loss / total_samples
        acc_train = accuracy_score(y_true, y_pred)
        precision_train = precision_score(y_true, y_pred, average='macro', zero_division=0.0)
        recall_train = recall_score(y_true, y_pred, average='macro', zero_division=0.0)
        F1_train = f1_score(y_true, y_pred, average='macro')
    
    return loss_train, acc_train, precision_train, recall_train, F1_train, y_pred, y_true, y_probs, galaxy_ids


def valid_epoch(model, data_loader, loss_func, device, coarse_labels=None, num_classes=7):
    total_loss = 0
    total_samples = 0
    y_true = []
    y_pred = []
    y_probs = []
    galaxy_ids = []

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            imgs = data[0]['data'].to(device)
            labels = data[0]['label'].to(device).view(-1).long()
            batch_galaxy_ids = data[0]['galaxy_id'].cpu().numpy()
            batch_size = labels.size(0)
            total_samples += batch_size

            with torch.autocast(device_type=device, dtype=torch.float16):
                if coarse_labels is not None:
                    # Convert coarse labels to one-hot encoding
                    batch_coarse = torch.tensor(coarse_labels[i*data_loader.batch_size:(i+1)*data_loader.batch_size])
                    batch_coarse = to_one_hot(batch_coarse, num_classes, device=device)
                    outputs = model(imgs, coarse_label=batch_coarse)
                else:
                    outputs = model(imgs)
                
                # Extract logits from CvTOutput
                logits = outputs.logits
                loss = loss_func(logits, labels)

            total_loss += loss.item() * batch_size

            probabilities = F.softmax(logits, dim=1)
            preds = probabilities.argmax(dim=1)
            
            y_true.extend(labels.detach().cpu().numpy().tolist())
            y_pred.extend(preds.detach().cpu().numpy().tolist())
            y_probs.extend(probabilities.detach().cpu().numpy().tolist())
            galaxy_ids.extend(batch_galaxy_ids.tolist())

            valid_loss = total_loss / total_samples
            valid_acc = accuracy_score(y_true, y_pred)
            valid_precision = precision_score(y_true, y_pred, average='macro', zero_division=0.0)
            valid_recall = recall_score(y_true, y_pred, average='macro', zero_division=0.0)
            valid_F1 = f1_score(y_true, y_pred, average='macro')
    
    return valid_loss, valid_acc, valid_precision, valid_recall, valid_F1, y_pred, y_true, y_probs, galaxy_ids


def train_model(n_epochs, model, train_loader, valid_loader, loss_func, optimizer, learning_scheduler, device, max_grad_norm, save_name:str='none', train_coarse=None, valid_coarse=None, num_classes=7, model_path='../output/'):
    model.to(device)

    results = {}
    results_class = {}
    time_start = time()
    trigger = 0
    patience = 35

    run = wandb.init(project='gmorph', name=save_name, 
                     config={'n_epochs': n_epochs, 
                             'batch_size': train_loader.batch_size, 
                             'lr': optimizer.param_groups[0]['lr'],
                             'optimizer': optimizer.__class__.__name__,
                             'scheduler': learning_scheduler.__class__.__name__ if learning_scheduler is not None else None,
                             'model': model.__class__.__name__})
    print("Run name:", run.name)

    for epoch in range(n_epochs):
        train_loss, train_acc, train_prec, train_recall, train_F1, train_pred, train_true, train_probs, train_galaxy_ids = train_epoch(model, optimizer, train_loader, loss_func, device, max_grad_norm, coarse_labels=train_coarse, num_classes=num_classes)

        run.log({'train_loss': train_loss, 'train_acc': train_acc, 'train_precision': train_prec, 'train_recall': train_recall, 'train_F1': train_F1})

        if learning_scheduler is not None:
            learning_scheduler.step()

        valid_loss, valid_acc, valid_prec, valid_recall, valid_F1, valid_pred, valid_true, valid_probs, valid_galaxy_ids = valid_epoch(model, valid_loader, loss_func, device, coarse_labels=valid_coarse, num_classes=num_classes)

        run.log({'valid_loss': valid_loss, 'valid_acc': valid_acc, 'valid_precision': valid_prec, 'valid_recall': valid_recall, 'valid_F1': valid_F1})

        print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - Valid Loss: {valid_loss:.4f} - Valid Acc: {valid_acc:.4f}")

        results[epoch] = [train_loss, train_acc, train_prec, train_recall, train_F1, valid_loss, valid_acc, valid_prec, valid_recall, valid_F1]

        precc = precision_score(train_true, train_pred, average=None, zero_division=0)
        recc = recall_score(train_true, train_pred, average=None, zero_division=0)
        f1c = f1_score(train_true, train_pred, average=None)

        preccv = precision_score(valid_true, valid_pred, average=None, zero_division=0)
        reccv = recall_score(valid_true, valid_pred, average=None, zero_division=0)
        f1cv = f1_score(valid_true, valid_pred, average=None)

        results_class[epoch] = [train_probs, train_true, train_galaxy_ids, precc, recc, f1c, valid_probs, valid_true, valid_galaxy_ids, preccv, reccv, f1cv]

        if epoch>0:
            if valid_loss>results[epoch-1][5]:
                trigger += 1
                if trigger>=patience:
                    print("Early stopping!")
                    print("EPOCH:",epoch)

                    if save_name:
                        torch.save(model.state_dict(), model_path+save_name+'.pth')
                        print(model_path + 'model_' + save_name, "is saved!")
                    
                    return results, results_class, train_pred, train_true, train_probs, train_galaxy_ids, valid_pred, valid_true, valid_probs, valid_galaxy_ids

        if save_name:
            filename = f"{model_path}model_{save_name}_epoch{epoch + 1}.pth"
            torch.save(model.state_dict(), filename)
            print(filename, "is saved!")
            
    run.finish()
    time_end = time()
    print("Training time =", (time_end - time_start) / 60, "minutes")

    return results, results_class, train_pred, train_true, train_probs, train_galaxy_ids, valid_pred, valid_true, valid_probs, valid_galaxy_ids


def test_model(dataset, model, device):
    out = torch.tensor([])
    y_true = torch.tensor([])
    y_preds = torch.tensor([])
    galaxy_ids = []

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(dataset):
            imgs = data[0]['data'].to(device)
            labels = data[0]['label'].to(device).view(-1).long()
            coarse_labels = data[0]['coarse_label'].to(device).view(-1).long()
            batch_galaxy_ids = data[0]['galaxy_id'].cpu().numpy()

            with torch.autocast(device_type=device, dtype=torch.float16):
                outputs = model(imgs, coarse_labels)

            probabilities = F.softmax(outputs, dim=1)
            preds = probabilities.argmax(dim=1)
            
            y_true = torch.cat((y_true, labels.detach().cpu()), 0)
            y_preds = torch.cat((y_preds, preds.detach().cpu()), 0)
            galaxy_ids.extend(batch_galaxy_ids.tolist())

    return y_true, y_preds, galaxy_ids