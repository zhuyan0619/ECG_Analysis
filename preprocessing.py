#Import packages
import numpy as np
import torchvision
import pandas as pd
import torch
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from torch.nn import Linear, ReLU,LeakyReLU, CrossEntropyLoss, Sequential, Conv1d, MaxPool1d, Module, Softmax, Flatten, BatchNorm1d
from torch.optim import Adam, SGD
from torch import no_grad
from torch.utils.data import Dataset, DataLoader

import pywt

import scipy

from tqdm.auto import tqdm


!pip --quiet install wfdb

!pip --quiet install matplotlib==3.1.3

import matplotlib.pyplot as plt

import os
import wfdb

#Installing the ECG recordings from the MIT BIH Arrhythmia Database onto Colab 
wfdb.dl_database('mitdb', os.path.join(os.getcwd(), 'mitdb'))

# Function to classify beat class

def classify_beat_class(symbol):
    if symbol in ["N", "R", "L", "e", "j"]: #Non-ectopic class according to AAMI standards
        return 0  
    elif symbol in ['S',"a", "A", "J"]:          #Supra-ventricular ectopic class
        return 1
    elif symbol in ['V', 'E', "!"]:              #Ventricular ectopic class
        return 2
    elif symbol =='F':                           #Fusion class
        return 3
    elif symbol in ['Q', '/', 'f']:              #Unknown class
        return 4

#Function classifies the beat type

def classify_beat(symbol):
    beat_types =["N", "R", "L", "e", "j",'S',"a", "A", "J", 'V', 'E', "!", 'F', 'Q', '/', 'f'] # ["N", "A", "V", "F", "/"] 
    for index, value in enumerate(beat_types):
      if symbol == value: 
        return index

# Function to return a sequence surrounding a beat 
# with window_size for each side

def get_sequence(signal, beat_loc, window_sec, fs):
    window_one_side = int(window_sec * fs)
    beat_start = beat_loc - window_one_side
    beat_end = beat_loc + window_one_side #+ 1
    if beat_end < signal.shape[0]:
        sequence = signal[beat_start:beat_end, 0]
        return sequence.reshape(1, -1, 1)
    else:
        return np.array([])
      
      
      
      
from sklearn.preprocessing import StandardScaler

all_sequences = []
all_labels = []
window_sec = 130/360

records = list(range(100, 110))+ list(range(111, 120)) +list(range(121, 125))+ [i for i in range(200,235) if (i not in [204, 206, 211, 216,218, 224,225,226,227,229])]
for subject in records:
    record = wfdb.rdrecord(f'mitdb/{subject}')
    annotation = wfdb.rdann(f'mitdb/{subject}', 'atr')
    atr_symbol = annotation.symbol
    atr_sample = annotation.sample

    fs = record.fs
    # Normalizing by mean and standard deviation
    scaler = StandardScaler()
    signal = scaler.fit_transform(record.p_signal)
    subject_labels = []
    for i, i_sample in enumerate(atr_sample):
        label = classify_beat(atr_symbol[i])
        sequence = get_sequence(signal, i_sample, window_sec, fs)
        if label is not None and sequence.size > 0:
            all_sequences.append(sequence)
            subject_labels.append(label)
    if len(subject_labels) == 0:
      continue
    all_labels.extend(subject_labels)
