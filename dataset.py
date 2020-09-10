import torch
import numpy as np 
from torch.utils import data
from torch.utils.data import DataLoader
import os 
import pickle
import glob

class KSSDatasetShuffle(torch.utils.data.Dataset):
    def __init__(self, train_path, batch_size):
        tmp = sorted(glob.glob(train_path + '*.npy'),reverse=True)
        self.batch_size = batch_size
        self.real_batch = 16
        tmp_list = []
        for i in range(len(tmp)//self.real_batch):
            tmp_list.append([tmp[i*self.real_batch:(i+1)*self.real_batch]])
        tmp_list.append([tmp[(len(tmp)//self.real_batch)*self.real_batch:]])
        self.train_list = tmp_list            
        
        
    def __len__(self):
        len_data = len(self.train_list)
        return len_data
    
    def __getitem__(self, index):
        data = np.load(self.train_list[index][0][0], allow_pickle=True)
        
        len_batch = len(self.train_list[index][0])
        
        text = []
        text_len = []
        mel = []
        stft = []
        mel_len = []
        pitch = []
        for i in range(len_batch):
            data = np.load(self.train_list[index][0][i], allow_pickle=True)
            text.append(data[0])
            text_len.append(data[1])
            mel.append(data[2])
            stft.append(data[3])
            mel_len.append(data[4])
            pitch.append(data[5])
        
        return (text, mel, text_len, mel_len, stft, pitch)
    
def my_collate(batch):
    text = [item[0] for item in batch]
    mel = [item[1] for item in batch]
    text_len = [item[2] for item in batch]
    mel_len = [item[3] for item in batch]
    stft = [item[4] for item in batch]
    pitch = [item[5] for item in batch]
    return [text, mel, text_len, mel_len, stft, pitch]