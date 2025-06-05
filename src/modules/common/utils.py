import yaml
import pprint
import os
import logging
import sys

import pandas as pd
import csv

import pickle
import numpy as np
import random 
import torch

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.DEBUG,
    stream=sys.stdout,
)

def load_cfg(yaml_filepath):
    
    ### Load a yaml configuration file.
    """
    # Parameters
        - yaml file path: str
    # Returns
        - cfg: dict
    """
    
    with open(yaml_filepath, "r") as f:
        cfg = yaml.safe_load(f)
        
    cfg = make_paths_absolute(os.path.dirname(yaml_filepath), cfg)
    
    return cfg
    
    #return Config(cfg)

def make_paths_absolute(dir_, cfg):
    
    ### Make all values for keys ending with '_path' absolute to dir_.
    
    """
    # Parameters
        - dir_: str
        - cfg: dict
    # Returns
        - cfg: dict
    """
    
    for key in cfg.keys():
        if key.endswith("_path"):
            cfg[key] = os.path.join(dir_, cfg[key])
            cfg[key] = os.path.abspath(cfg[key])
            if not os.path.exists(cfg[key]):
                logging.warning("%s does not exist.", cfg[key])
        if type(cfg[key]) is dict:
            cfg[key] = make_paths_absolute(dir_, cfg[key])
            
    return cfg

def logging_result(result):
    
    logging.info('----- type: %s', result['type'])
    logging.info('----- fold: %s', result['fold'])
    logging.info('----- auroc: %f', result['auroc'])
    logging.info('----- auprc: %f', result['auprc'])
    logging.info('----- f1: %f', result['f1'])
    logging.info('----- precision: %f', result['precision'])
    logging.info('----- recall: %f', result['recall'])
    logging.info('----- accuracy: %f', result['accuracy'])
    logging.info('----- mcc: %f', result['mcc'])
    
    return

def cal_acc(pred, target, ignore_idx=None):
    
    if ignore_idx is not None:
        mask = torch.ones_like(target).scatter_(0, ignore_idx.long(), 0.0)
    else:
        mask = torch.ones_like(target)
        
    masked_target = target.masked_select(mask.bool())
    masked_pred = pred.masked_select(mask.bool())
    
    acc = torch.sum(masked_target == masked_pred).float() / len(masked_target)
    
    return acc

def cal_metrics_forecast(pred, target, weight):
    
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    weight = weight.cpu().numpy()
    
    rmse = cal_rmse(pred, target, weight)
    mae = cal_mae(pred, target, weight)
    corr = cal_corr(pred, target, weight)
    ci = cal_ci(pred, target, weight)
    
    return rmse, mae, corr, ci

def cal_rmse(pred, target, weight):
    
    weight_ = np.reshape(weight, (-1))
    squared_error_all = np.square(pred - target)
    squared_error_all = np.reshape(squared_error_all, (-1))
    
    score = np.sqrt(np.sum(squared_error_all * weight_) / np.sum(weight_))
    
    return score

def cal_mae(pred, target, weight):
    
    weight_ = np.reshape(weight, (-1))
    error_all = np.abs(pred - target)
    error_all = np.reshape(error_all, (-1))
    
    score = np.sum(error_all * weight_) / np.sum(weight_)
    
    return score

def cal_corr(pred, target, weight):
    
    weight_ = np.reshape(weight, (-1))
    target_ = np.reshape(target, (-1))
    pred_ = np.reshape(pred, (-1))
    
    idxs = np.where(weight_ > 0.5)[0]
    
    target_ = target_[idxs]
    pred_ = pred_[idxs]
    
    if np.var(target_) < 1e-10 or np.var(pred_) < 1e-10:
        return 0.0
    
    score = np.corrcoef(target_, pred_)[0,1]
    
    return score 

def cal_ci(pred, target, weight):
    
    weight_ = np.reshape(weight, (-1))
    target_ = np.reshape(target, (-1))
    pred_ = np.reshape(pred, (-1))
    
    idxs = np.where(weight_ > 0.5)[0]
    
    if len(idxs) <= 1:
        return 0.0
    
    target_ = target_[idxs]
    pred_ = pred_[idxs]
    
    score = cal_ci_(pred_, target_)
    
    return score

def cal_ci_(pred, target):
    
    idxs = np.argsort(target)
    tar_srt = target[idxs]
    pre_srt = pred[idxs]
    
    ci = 0
    num = len(tar_srt)
    
    if num <= 1:
        return 0.0 
    else:
        correct_pairs = 0
        total_pairs = 0
        for i in range(num-1):
            for j in range(i+1, num):
                if tar_srt[i] < tar_srt[j]:
                    total_pairs += 1
                    if pre_srt[i] < pre_srt[j]:
                        correct_pairs += 1
    
    ci = correct_pairs / total_pairs
    
    return ci

def cal_ci_with_idx(pred, target, idx_pairs):
    
    pred = np.reshape(pred, (-1))
    target = np.reshape(target, (-1))
    
    print(pred.shape, target.shape, idx_pairs.shape)
    
    correct_pairs = 0
    
    for i in range(idx_pairs.shape[0]):
        lower_idx = idx_pairs[i, 0]
        upper_idx = idx_pairs[i, 1]
        
        if target[lower_idx] >= target[upper_idx]:
            continue
        
        if pred[lower_idx] < pred[upper_idx]:
            correct_pairs += 1
    
    ci = correct_pairs / idx_pairs.shape[0]
    
    return ci
