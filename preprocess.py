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

import amfm_decompy.pYAAPT as pYAAPT
import amfm_decompy.basic_tools as basic


def tnorm(txt):
  txt = txt.lower()
  txt = re.sub("[\"\-()[\]“”]", " ", txt)
  txt = re.sub("[,;:!'?’]", ".", txt)
  return txt


sample = []

nameNsize = []

with codecs.open("../kss/text.txt", 'r', "utf-8") as f:
  lines = f.readlines()
  for line in lines:
    line = line.strip()
    fname = line.split('|')[0]
    line = line.split('|')[2]
    fsize = os.path.getsize('../kss/{}'.format(fname))
    nameNsize.append((fname,line,fsize))

  
nameNsize.sort(key = lambda element : element[2])
lendata = len(nameNsize)
print(lendata)

import matplotlib.pyplot as plt 

import amfm_decompy.pYAAPT as pYAAPT
import amfm_decompy.basic_tools as basic

def preprocessing(ii):
    fname = nameNsize[ii][0]
    line = nameNsize[ii][1]
    phone = []
    for item in line.split(' '):
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
    dic = ['P','.','!','?',',','k0','kk','nn','t0','tt','rr','mm','p0','pp','s0','ss','oh','c0','cc','ch','kh','th','ph','h0','aa','qq','ya','yq','vv','ee','yv','ye','oo','wa','wq','wo','yo','uu','wv','we','wi','yu','xx','xi','ii','','kf','ks','nf','nc','ng','nh','tf','ll','lk','lm','lb','ls','lt','lp','lh','mf','pf','ps',' ','E']
    char2idx = {ch: idx for idx, ch in enumerate(dic)}
    emblen = len(char2idx)
    txt = np.asarray([char2idx[ch] for ch in text])
    audio, sr = librosa.load('../kss/{}'.format(fname), sr = 22050)
    audio, index = librosa.effects.trim(audio, top_db=43, frame_length=256, hop_length=64)
    audioobj = basic.SignalObj(audio,sr)
    pitch = pYAAPT.yaapt(audioobj,**{'f0_min' : 100.0, 'frame_length' : 1000*1024//22050, 'frame_space' : 1000*256//22050})
    pitch = pitch.samp_values
    stft = np.abs(librosa.stft(audio,n_fft=1024,hop_length=256,win_length=1024))
    stft = np.power(stft/np.max(stft), 0.6)
    mel_filters = librosa.filters.mel(22050,1024,80)
    mel = np.dot(mel_filters,stft)
    mel = np.power(mel/np.max(mel), 0.6)
    mel = mel[:,::4]
    pitch = pitch[::4]
    length = np.min([np.shape(mel)[1], np.shape(stft)[1]//4])
    mel = mel[:,:length]
    stft = stft[:,:length*4]
    pitch = pitch[:length]
    np.save('data/sample_{}.npy'.format((str)(ii).zfill(5)),(txt,len(txt),mel,stft,mel.shape[1],pitch))
    print('\r{}saved'.format(ii),end='')

import multiprocessing


if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=30)
    pool.map(preprocessing, np.arange(12854))
    pool.close()
    pool.join()
