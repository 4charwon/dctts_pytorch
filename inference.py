import numpy as np
import torch
import torch.nn as nn
import librosa
import cv2
import matplotlib.pyplot as plt
from torch.nn import utils
from g2p_module import G2P
import argparse

import sys
import os

import torch.nn.functional as F
from torch.autograd import Variable
np.random.seed(0)

torch.manual_seed(123)
torch.cuda.manual_seed(123)

from util import spectrogram2wav

from networks import *
from dataset import *

syn_dir_t2m = 'result_t2m'#kss_result3'
syn_dir_ssrn = 'result_ssrn'

def spectrogram2wav(srstft):
  srstft = srstft.cpu().detach().numpy()
  recon = librosa.istft(np.random.rand(srstft.shape[0],srstft.shape[1]),hop_length=256,win_length=1024)
  for i in range(100):
    recon_stft = librosa.stft(recon,n_fft=1024,hop_length=256,win_length=1024)
    recon_angle = np.angle(recon_stft)
    new_guess = srstft*np.exp(1.0j*recon_angle)
    recon = librosa.istft(new_guess,hop_length=256,win_length=1024)
  return recon

graph0 = Text2Mel()
graph0 = nn.DataParallel(graph0)
graph1 = SSRN()
graph0.load_state_dict(torch.load(syn_dir_t2m+'/t2m180.pth'))
graph1.load_state_dict(torch.load(syn_dir_ssrn+'/sr100.pth'))
graph0.eval()
graph1.eval()
graph0.cuda()
graph1.cuda()


def inference():
    parser = argparse.ArgumentParser(description="optional actions")
    parser.add_argument("--text", type=str, default = '동해물과 백두산이')
    args = parser.parse_args()
    text_seq = args.text
    
    texts = []
    dic  =['P','.','!','?',',','k0','kk','nn','t0','tt','rr','mm','p0','pp','s0','ss','oh','c0','cc','ch','kh','th','ph','h0','aa','qq','ya','yq','vv','ee','yv','ye','oo','wa','wq','wo','yo','uu','wv','we','wi','yu','xx','xi','ii','','kf','ks','nf','nc','ng','nh','tf','ll','lk','lm','lb','ls','lt','lp','lh','mf','pf','ps',' ','E']
    char2idx = {ch: idx for idx, ch in enumerate(dic)}
    idx2char = {idx: ch for idx, ch in enumerate(dic)}
    emblen = len(char2idx)
    phone = []
    for item in text_seq.split(' '):
          temp = G2P(item)
          phone += temp.split()
          if item.find('!') != -1 :
            phone += '!'
          elif item.find('?') != -1 :
            phone += '?'
          elif item.find('.') != -1 :
            phone += '.'
          elif item.find(',') != -1 :
            phone += ','
          phone += ' '
    text = phone[:-1] + ['E']
    text = np.asarray([char2idx[ch] for ch in text])
    reverse = np.asarray([idx2char[idx] for idx in text])
    texts.append(text)
    texts = np.asarray(texts)
    texts = torch.from_numpy(texts).type(torch.LongTensor).cuda()
    mels = torch.FloatTensor(np.zeros((1,80,1))).cuda()
    for t in range(texts.size()[1]*2):
      new_mel, prev_atten = graph0.module(texts,mels)
      mels = torch.cat([mels,new_mel[:,:,-1:]],dim=2)

    mag = graph1(mels)

    plt.figure(figsize=(16,9))
    plt.subplot(3,1,2)
    plt.imshow(np.log(mels[0].detach().cpu().numpy()+1e-5))
    plt.subplot(3,1,3)
    plt.imshow(np.log(mag[0].detach().cpu().numpy()+1e-5))
    plt.subplot(3,1,1)
    plt.imshow(prev_atten[0].detach().cpu().numpy())

    plt.savefig('inference.png')
    plt.close()

    srstft = mag[0]**(1.3/0.6)
    recon = spectrogram2wav(srstft)
    librosa.output.write_wav(text_seq+'.wav', recon/(1.1*np.max(np.abs(recon))), 22050)
if __name__ == "__main__":
    inference()
