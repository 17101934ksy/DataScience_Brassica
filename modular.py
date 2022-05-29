import numpy as np
import matplotlib.pyplot as plt

import random
import os
import math
import pickle

import pathlib
from pathlib import Path
from glob import glob

import pandas as pd
import cv2
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR, CyclicLR, CosineAnnealingWarmRestarts
from torchsummary import summary

import torchvision.models as models
from torchvision import transforms
from torchvision.transforms.transforms import Resize, CenterCrop, Lambda, RandomHorizontalFlip, RandomRotation, RandomVerticalFlip

from collections import OrderedDict

from sklearn.preprocessing import StandardScaler

import math

# import warnings
# warnings.filterwarnings(action='ignore')



CFG = {
    'IMG_SIZE': 224,
    'EPOCHS': 100,
    'LEARNING_RATE': 0.005,
    'BATCH_SIZE': 128,
    'SEED': 41,
    'EPS' : 1e-12
}


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CFG['SEED']) # Seed 고정
