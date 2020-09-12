import numpy as np
import torch
import torch.nn as nn
import librosa
import cv2
import matplotlib.pyplot as plt
from torch.nn import utils
from g2p_module import G2P

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

graph1 = SSRN()

batch_size = 1

syn_dir = 'result_ssrn'
if not os.path.exists(syn_dir):
  os.makedirs(syn_dir)

graph1.cuda()
sr_opt = torch.optim.Adam(graph1.parameters(),lr=0.0002,betas=(0.5, 0.9), eps=1e-6)
graph1.load_state_dict(torch.load('result_ssrn/sr100.pth'))

dataset = KSSDatasetShuffle('data/', batch_size)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=my_collate)
t2m_opt = torch.optim.Adam(graph1.parameters(),lr=0.0002,betas=(0.5, 0.9), eps=1e-6)#,amsgrad=True)

mel_loss_plot=[]
for ep in range(101,10000):
    itera = -1
    for i, data in enumerate(train_loader):
      if i % 4 == 0 :
        print('\r/',end='')
      if i % 4 == 1 :
        print('\r-',end='')
      if i % 4 == 2 :
        print('\r\\',end='')
      if i % 4 == 3 :
        print('\r|',end='')
      
      itera = itera+1
      _, mel, _, mel_len, stft ,_ = data
      
      mel = mel[0]
      mel_len = mel_len[0]
      stft = stft[0]
      
      temp_batch_size = len(mel)
    
      b_stft = np.zeros((temp_batch_size,513,4*np.max(mel_len)))
      b_mel = np.zeros((temp_batch_size,80,np.max(mel_len)))
    
      for k in range(temp_batch_size):
        b_stft[k,:,:np.shape(stft[k])[1]] = stft[k]
        b_mel[k,:,:np.shape(mel[k])[1]] = mel[k]
        
      target_stft = torch.Tensor(b_stft).cuda()
      target_mel = torch.Tensor(b_mel).cuda()
        
      sr_opt.zero_grad()
      mags = graph1(target_mel)
      
      l1_loss = torch.mean(torch.abs(mags-target_stft))
      bin_div = torch.mean(-target_stft * torch.log(mags+1e-8) - (1-target_stft)*torch.log(1-mags+1e-8)) - torch.mean(-target_stft * torch.log(target_stft+1e-8) - (1-target_stft)*torch.log(1-target_stft+1e-8))
      mag_loss = l1_loss+bin_div
      mag_loss.backward()
    
      sr_opt.step()
      
      if itera % 300 == 0 :
          print('\n')
          print('epoch #'+str(ep)+' data #'+str(itera)+'\n mag_loss : '+str(mag_loss.item())+' l1_loss : '+str(l1_loss.item())+' bin : '+str(bin_div.item()))
          plt.figure(figsize=(16,9))
          plt.subplot(3,1,1)
          plt.imshow(np.log(target_mel[0].detach().cpu().numpy()+2**-4))
          plt.subplot(3,1,2)
          plt.imshow(np.log(target_stft[0].detach().cpu().numpy()+2**-4))
          plt.subplot(3,1,3)
          plt.imshow(np.log(mags[0].detach().cpu().numpy()+2**-4))
          plt.savefig(syn_dir+'/mel_log_{}_{}.png'.format(ep,itera))
          mel_loss_plot.append(mag_loss.detach().cpu())
          plt.close("all")
    
    plt.figure()
    plt.plot(mel_loss_plot)
    plt.title('Mel Loss')
    plt.savefig(syn_dir+'/loss_{}.png'.format(ep))
    plt.close("all")
    
    if ep % 10 == 0:
      torch.save(graph1.state_dict(),syn_dir+'/sr{}.pth'.format(ep))
