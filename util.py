import sys
import os
import torch
torch.manual_seed(0)
import torchvision
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
from torch.nn import utils
import soundfile as sf
from g2p_module import G2P
torch.manual_seed(123)
torch.cuda.manual_seed(123)

import wave
import torch
import librosa
import librosa.display
from IPython.display import Audio, display
import numpy as np
import scipy
import codecs
import re


def spectrogram2wav(srstft):
  srstft = srstft.cpu().detach().numpy()
  recon = librosa.istft(np.random.rand(srstft.shape[0],srstft.shape[1]),hop_length=256,win_length=1024);
  length = recon.shape
  for i in range(0,200):
    recon_stft = librosa.stft(recon,n_fft=1024,hop_length=256,win_length=1024);
    recon_angle = np.angle(recon_stft);
    new_guess = srstft*np.exp(1.0j*recon_angle);
    recon = librosa.istft(new_guess,hop_length=256,win_length=1024);
    
  return recon


def validation(text_seq,emb_mel=None):
  graph0.eval()
  texts = []
  dic  =['P','.','!','?',',','k0','kk','nn','t0','tt','rr','mm','p0','pp','s0','ss','oh','c0','cc','ch','kh','th','ph','h0','aa','qq','ya','yq','vv','ee','yv','ye','oo','wa','wq','wo','yo','uu','wv','we','wi','yu','xx','xi','ii','','kf','ks','nf','nc','ng','nh','tf','ll','lk','lm','lb','ls','lt','lp','lh','mf','pf','ps',' ','E']
  char2idx = {ch: idx for idx, ch in enumerate(dic)}
  idx2char = {idx: ch for idx, ch in enumerate(dic)}
  EMBLEN = len(char2idx)
  phone = []
  for item in text_seq.split(' '):
    temp = G2P(item)
    phone += temp.split()
    phone += ' '
  text = phone[:-1] + ['E']
  text = np.asarray([char2idx[ch] for ch in text])
  reverse = np.asarray([idx2char[idx] for idx in text])
  texts.append(text)
  texts = np.asarray(texts)
  texts = torch.from_numpy(texts).type(torch.LongTensor).cuda()
  mels = torch.FloatTensor(np.zeros((1,80,1))).cuda()
  for t in range(50):
    new_mel, attention = graph0.module(texts,mels)
    mels = torch.cat([mels,new_mel[:,:,-1:]],dim=2)
    
  plt.figure(figsize=(30,10))
  plt.subplot(1,3,1)
  plt.imshow(attention[0].detach().cpu().numpy())
  plt.subplot(1,3,2)
  plt.imshow(np.log(mels[0].detach().cpu().numpy()+2**-4),aspect=1/4)
  plt.savefig('validation_{}.png'.format(text_seq))
  plt.close()
  graph0.train()
  from networks import SuperRes
  graph1 = SuperRes()
  graph1.cuda()
  #graph1.load_state_dict(torch.load('result_ssrn/sr50.pth'))
  graph1.load_state_dict(torch.load('kss_ssrn_result/t2m270.pth'))
  _,mag = graph1(mels)
  recon = spectrogram2wav(mag[0]**(1.3/0.6))
  librosa.output.write_wav(text_seq+'.wav', recon/(1.1*np.max(np.abs(recon))), 22050)

  return mels, attention

def guided_attention(batch_size, text_length, audio_length, g):
    guided_attention_matrix = np.zeros((batch_size, text_length, audio_length), dtype='float32')
    for i in range(text_length):
        for j in range(audio_length):
            guided_attention_matrix[:,i,j] = 1-np.exp((-(i/text_length - j/audio_length)*(i/text_length - j/audio_length))/(2*g*g))
    return guided_attention_matrix