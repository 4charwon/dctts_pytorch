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

graph0 = Text2Mel()
graph0 = nn.DataParallel(graph0)
graph0.load_state_dict(torch.load('result_t2m/t2m180.pth'))
graph0.cuda()

batch_size = 1

syn_dir = 'result_t2m'
if not os.path.exists(syn_dir):
  os.makedirs(syn_dir)

dataset = KSSDatasetShuffle('../1prosodytts/data_rs/', batch_size)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=my_collate)
t2m_opt = torch.optim.Adam(graph0.parameters(),lr=0.0002,betas=(0.5, 0.9), eps=1e-6)#,amsgrad=True)

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
  for t in range(100):
    new_mel, attention = graph0.module(texts,mels,emb_mel)
    mels = torch.cat([mels,new_mel[:,:,-1:]],dim=2)
    
  plt.figure(figsize=(30,10))
  plt.subplot(1,3,1)
  plt.imshow(attention[0].detach().cpu().numpy())
  plt.subplot(1,3,2)
  plt.imshow(np.log(mels[0].detach().cpu().numpy()+2**-4),aspect=1/4)
  plt.savefig('validation_{}.png'.format(text_seq))
  plt.close()
  graph0.train()
  graph1 = SSRN()
  graph1.cuda()
  graph1.load_state_dict(torch.load('result_ssrn/sr100.pth'))
  mag = graph1(mels)
  recon = spectrogram2wav(mag[0]**(1.3/0.6))
  librosa.output.write_wav(text_seq+'.wav', recon/(1.1*np.max(np.abs(recon))), 22050)

  return mels, attention


def guided_attention(batch_size, text_length, audio_length, g):
    guided_attention_matrix = np.zeros((batch_size, text_length, audio_length), dtype='float32')
    for i in range(text_length):
        for j in range(audio_length):
            guided_attention_matrix[:,i,j] = 1-np.exp((-(i/text_length - j/audio_length)*(i/text_length - j/audio_length))/(2*g*g))
    return guided_attention_matrix

mel_loss_plot=[]
for ep in range(181,10000):
    itera = -1
    validation('동해물과 백두산이')
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
      text, mel, text_len, mel_len, _ , _ = data
      text = text[0]
      mel = mel[0]
      text_len = text_len[0]
      mel_len = mel_len[0]
      temp_batch_size = len(text)
        
      b_txt = np.zeros((temp_batch_size,np.max(text_len)))
      b_mel = np.zeros((temp_batch_size,80,np.max(mel_len)))
        
      for k in range(temp_batch_size):
        b_txt[k,:len(text[k])] = text[k]
        b_mel[k,:,:np.shape(mel[k])[1]] = mel[k]
        
      texts = torch.LongTensor(np.asarray(b_txt)).cuda()
      target_mel = torch.Tensor(b_mel).cuda()
      input_mel = torch.cat([torch.zeros(target_mel.size()[0],target_mel.size()[1],1).type(torch.FloatTensor).cuda(),target_mel[:,:,:-1]],dim=2)
      
      t2m_opt.zero_grad()
    
      mel, atten = graph0(texts, input_mel)
        
      l1_loss = torch.mean(torch.abs(mel-target_mel))
      bin_div = torch.mean(-target_mel * torch.log(mel+1e-8) - (1-target_mel)*torch.log(1-mel+1e-8)) - torch.mean(-target_mel * torch.log(target_mel+1e-8) - (1-target_mel)*torch.log(1-target_mel+1e-8))
      w = guided_attention(temp_batch_size, np.shape(b_txt)[1], np.shape(b_mel)[2], 0.2)
      w = torch.Tensor(w).cuda()
      att_loss = torch.mean(w*atten)
      mel_loss = l1_loss+bin_div+att_loss
      
      mel_loss.backward()
      
      t2m_opt.step()
      
      if itera % 300 == 0 :
          print('\n')
          print('epoch #'+str(ep)+' data #'+str(itera)+'\n mel_loss : '+str(mel_loss.item())+'att_loss : '+str(att_loss.item())+'l1 : '+str(l1_loss.item())+'bd : '+str(bin_div.item()))
          plt.figure(figsize=(30,10))
          plt.subplot(1,3,1)
          plt.imshow(atten[0].detach().cpu().numpy(),aspect=1/2)
          plt.subplot(1,3,2)
          plt.imshow(np.log(mel[0].detach().cpu().numpy()+2**-4),aspect=1/4)
          plt.subplot(1,3,3)
          plt.imshow(np.log(target_mel[0].detach().cpu().numpy()+2**-4),aspect=1/4)
          plt.savefig(syn_dir+'/mel_log_{}_{}.png'.format(ep,itera))
          mel_loss_plot.append(mel_loss.detach().cpu())
          plt.close("all")

    plt.figure(figsize=(20,10))
    plt.plot(mel_loss_plot)
    plt.title('Mel Loss')
    plt.savefig(syn_dir+'/loss_{}.png'.format(ep))
    plt.close("all")
   
          
    if ep % 10 == 0:
      torch.save(graph0.state_dict(),syn_dir+'/t2m{}.pth'.format(ep))
