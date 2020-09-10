import numpy as np
import torch
import torch.nn as nn
import librosa
import matplotlib.pyplot as plt

import sys
import os

import torch.nn.functional as F
from torch.autograd import Variable
np.random.seed(0)
from torch.nn import utils
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
import glob 

from util import spectrogram2wav

EncoderDimension = 256
EmbDimension = 128
NFFT = 1024
dic  =['P','.','!','?',',','k0','kk','nn','t0','tt','rr','mm','p0','pp','s0','ss','oh','c0','cc','ch','kh','th','ph','h0','aa','qq','ya','yq','vv','ee','yv','ye','oo','wa','wq','wo','yo','uu','wv','we','wi','yu','xx','xi','ii','','kf','ks','nf','nc','ng','nh','tf','ll','lk','lm','lb','ls','lt','lp','lh','mf','pf','ps',' ','E']
char2idx = {ch: idx for idx, ch in enumerate(dic)}
idx2char = {idx: ch for idx, ch in enumerate(dic)}
EMBLEN = len(char2idx)

from collections import OrderedDict

class SequentialMaker:
    def __init__(self):
        self.dict = OrderedDict()

    def add_module(self, name, module):
        if hasattr(module, "weight"):
            module = nn.utils.weight_norm(module)
        self.dict[name] = module

    def __call__(self):
        return nn.Sequential(self.dict)
  

class CausalHighwayConv(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size,dilation=1):
        self.kernel_size = kernel_size
        self.dilation = dilation
        super(CausalHighwayConv, self).__init__(in_channels, out_channels, kernel_size=kernel_size,dilation=dilation, padding = 0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        padding = (self.kernel_size[0]-1)*self.dilation[0]
        if padding != 0 :
            pad_inputs = torch.cat([torch.zeros(inputs.size()[0],inputs.size()[1],padding).type(torch.FloatTensor).cuda(),inputs],dim=2)
        h = super(CausalHighwayConv, self).forward(pad_inputs)
        h1, h2 = torch.chunk(h, 2, 1)
        h1 = self.sigmoid(h1)
        return h1 * h2 + (1-h1)*inputs
    
class NonCausalHighwayConv(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size,dilation=1,padding=0):
        super(NonCausalHighwayConv, self).__init__(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation, padding = padding)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        h = super(NonCausalHighwayConv, self).forward(inputs)
        h1, h2 = torch.chunk(h, 2, 1)  # chunk at the feature dim
        h1 = self.sigmoid(h1)
        return h1 * h2 + (1-h1)*inputs
        
class TextEnc(nn.Module):
  def __init__(self):
    super(TextEnc, self).__init__()
    self.emb = nn.Embedding(EMBLEN, 128,0)
    
    module = SequentialMaker()
    module.add_module('Conv1', nn.Conv1d(EmbDimension,EncoderDimension*2,kernel_size=1,dilation=1,padding=0))
    module.add_module('Relu',nn.ReLU())
    module.add_module('Dropout0',nn.Dropout(0.05))
    module.add_module('Conv2', nn.Conv1d(EncoderDimension*2,EncoderDimension*2,kernel_size=1,dilation=1,padding=0))
    module.add_module('Dropout1',nn.Dropout(0.05))
    
    kernel_size=3
    for j in range(2):
        for i in range(4):
            kernel_size=3
            dilation=3**i
            padding = ((kernel_size-1)*dilation+1)//2
            module.add_module('NCConv{}_{}'.format(i,j+1),NonCausalHighwayConv(EncoderDimension*2,EncoderDimension*4,kernel_size=3,dilation=3**i,padding=padding))
            module.add_module('Dropout{}_{}'.format(i,j+1),nn.Dropout(0.05))
    for j in range(2):
        for i in range(2):
            kernel_size=3**(1-j)
            dilation=1
            padding = ((kernel_size-1)*dilation+1)//2
            module.add_module('NCConv{}_{}'.format(i+4,j+1),NonCausalHighwayConv(EncoderDimension*2,EncoderDimension*4,kernel_size=3**(1-j),dilation=1,padding=padding))
            if j != 1 and i != 1:
                module.add_module('Dropout{}_{}_2'.format(i,j+1),nn.Dropout(0.05))
    self.module = module()
    
  def forward(self, inputs):
    x = self.emb(inputs)
    x = torch.transpose(x, 1,2)
    x = self.module(x)
    
    k, v = torch.chunk(x,2,1)
 
    return k, v

class AudioEnc(nn.Module):
  def __init__(self):
    super(AudioEnc, self).__init__()
    
    module = SequentialMaker()
    module.add_module('Conv1', nn.Conv1d(80,EncoderDimension,kernel_size=1,dilation=1,padding=0))
    module.add_module('LR1', (nn.LeakyReLU()))
    module.add_module('Dropout0',nn.Dropout(0.05))
    module.add_module('Conv2', nn.Conv1d(EncoderDimension,EncoderDimension,kernel_size=1,dilation=1,padding=0))
    module.add_module('LR2', (nn.LeakyReLU()))
    module.add_module('Dropout1',nn.Dropout(0.05))
    module.add_module('Conv2_2', nn.Conv1d(EncoderDimension,EncoderDimension,kernel_size=1,dilation=1,padding=0))
    module.add_module('LR3', (nn.LeakyReLU()))
    module.add_module('Dropout2',nn.Dropout(0.05))
    
    kernel_size=3
    for j in range(2):
        for i in range(4):
            module.add_module('CConv{}_{}'.format(i+3,j+1),CausalHighwayConv(EncoderDimension,EncoderDimension*2,kernel_size=3,dilation=3**i))
            module.add_module('Dropout{}_{}'.format(i+3,j+1),nn.Dropout(0.05))
    for i in range(2):
        module.add_module('CConv{}_{}'.format(7,i),CausalHighwayConv(EncoderDimension,EncoderDimension*2,kernel_size=3,dilation=3))
        if i != 1:
            module.add_module('Dropout{}_{}'.format(7,i),nn.Dropout(0.05))
    self.module = module()
    
  def forward(self, inputs):
    q = self.module(inputs)
 
    return q
    
class AudioDec(nn.Module):
  def __init__(self):
    super(AudioDec, self).__init__()
    
    module = SequentialMaker()
    module.add_module('Conv1', nn.Conv1d(EncoderDimension*2,EncoderDimension,kernel_size=1,dilation=1,padding=0))
    module.add_module('Dropout0',nn.Dropout(0.05))

    kernel_size=1
    dilation=1
    padding = ((kernel_size-1)*dilation+1)//2
    module.add_module('Conv2', NonCausalHighwayConv(EncoderDimension,EncoderDimension*2,kernel_size=1,dilation=1,padding=padding))
    module.add_module('Dropout1',nn.Dropout(0.05))
    
    kernel_size=3
    for j in range(2):
        for i in range(4):
            module.add_module('CConv{}_{}'.format(i+3,j+1),CausalHighwayConv(EncoderDimension,EncoderDimension*2,kernel_size=3,dilation=3**i))
            module.add_module('Dropout{}_{}'.format(i+3,j+1),nn.Dropout(0.05))
    for i in range(3):
        module.add_module('CConv{}_{}'.format(7,i),CausalHighwayConv(EncoderDimension,EncoderDimension*2,kernel_size=3,dilation=1))
        module.add_module('Dropout{}_{}'.format(7,i),nn.Dropout(0.05))
    module.add_module('Conv3',nn.Conv1d(EncoderDimension,80,kernel_size=1,dilation=1,padding=0))
    self.module = module()
    
  def forward(self, inputs):
    mel = self.module(inputs)
 
    return mel

class Text2Mel(nn.Module):
  def __init__(self):
    super(Text2Mel,self).__init__()
    self.texts_enc = TextEnc()
    self.audio_enc = AudioEnc()
    self.audio_dec = AudioDec()
    self.sigmoid = nn.Sigmoid()
  def forward(self, texts, shift_mels,emb_mel=None):
    k, v = self.texts_enc(texts)
    q = self.audio_enc(shift_mels)
    a = F.softmax(torch.bmm(k.transpose(1, 2), q) / np.sqrt(256), 1)
    r = torch.bmm(v,a)
    rp = torch.cat((r,q),1)
    self.mels = self.sigmoid(self.audio_dec(rp))
    self.attention = a
    
    return self.mels, self.attention
    
class SSRN(nn.Module):
  def __init__(self):
    super(SSRN, self).__init__()
    
    module = SequentialMaker()
    module.add_module('Conv1', nn.Conv1d(80,NFFT//2,kernel_size=1,dilation=1,padding=0))
    module.add_module('Dropout0',nn.Dropout(0.05))
    
    kernel_size=3
    for j in range(3):
        for i in range(2):
            kernel_size=3
            dilation=3**i
            padding = ((kernel_size-1)*dilation+1)//2
            module.add_module('NCConv{}_{}'.format(i+2,j+1),NonCausalHighwayConv(NFFT//2,NFFT,kernel_size=3,dilation=3**i,padding=padding))
            module.add_module('Dropout{}_{}'.format(i+2,j+1),nn.Dropout(0.05))
        if j != 2 :
            module.add_module('DeConv{}_{}'.format(i+2,j+1),nn.ConvTranspose1d(NFFT//2,NFFT//2,kernel_size=2,dilation=1,stride=2,padding=0))
            module.add_module('Dropout{}_{}_2'.format(i+2,j+1),nn.Dropout(0.05))
    module.add_module('Conv4',nn.Conv1d(NFFT//2,NFFT//2,kernel_size=1,dilation=1,padding=0))
    module.add_module('Dropout4',nn.Dropout(0.05))
    kernel_size=3
    dilation=1
    padding = ((kernel_size-1)*dilation+1)//2
    module.add_module('Conv5',NonCausalHighwayConv(NFFT//2,NFFT,kernel_size=3,dilation=1,padding=padding))
    module.add_module('Dropout5',nn.Dropout(0.05))
    kernel_size=3
    dilation=1
    padding = ((kernel_size-1)*dilation+1)//2
    module.add_module('Conv5_2',NonCausalHighwayConv(NFFT//2,NFFT,kernel_size=3,dilation=1,padding=padding))
    module.add_module('Dropout5_2',nn.Dropout(0.05))
    module.add_module('Conv6',nn.Conv1d(NFFT//2,NFFT//2+1,kernel_size=1,dilation=1,padding=0))
    module.add_module('Dropout4',nn.Dropout(0.05))
    module.add_module('Conv7',nn.Conv1d(NFFT//2+1,NFFT//2+1,kernel_size=1,dilation=1))
    module.add_module('Relu7',nn.ReLU())
    module.add_module('Dropout7',nn.Dropout(0.05))
    module.add_module('Conv7_2',nn.Conv1d(NFFT//2+1,NFFT//2+1,kernel_size=1,dilation=1))
    module.add_module('Relu7_2',nn.ReLU())
    module.add_module('Dropout7_2',nn.Dropout(0.05))
    module.add_module('Conv7_3',nn.Conv1d(NFFT//2+1,NFFT//2+1,kernel_size=1,dilation=1))
    self.Sigmoid = nn.Sigmoid()
    self.module = module()

  def forward(self, inputs):
    self.mag = self.Sigmoid(self.module(inputs))
 
    return self.mag