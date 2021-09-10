# Copyright (C) 2019 Computational Science Lab, UPF <http://www.compscience.org/>
# Copying and distribution is allowed under AGPLv3 license

import os
import sys
import torch
import torch.autograd
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
import numpy as np
from torchvision import transforms
from torchvision import datasets
from torchvision.utils import save_image

from networks import EncoderCNN, DecoderRNN, generator, discriminator
from generators import queue_datagen
from keras.utils.data_utils import GeneratorEnqueuer
from tqdm import tqdm, trange
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True, help="Path to input .smi file.")
parser.add_argument("-o", "--output_dir", required=True, help="Path to model save folder.")
args = vars(parser.parse_args())

cap_loss = 0.
caption_start = 4000
batch_size = 128

savedir = args["output_dir"]
os.makedirs(savedir, exist_ok=True)
smiles = np.load(args["input"])

import multiprocessing
multiproc = multiprocessing.Pool(6)
my_gen = queue_datagen(smiles, batch_size=batch_size, mp_pool=multiproc)
mg = GeneratorEnqueuer(my_gen, seed=0)
mg.start()
mt_gen = mg.get()

# Define the networks
encoder = EncoderCNN(8)
decoder = DecoderRNN(512, 1024, 29, 1)
D = discriminator(nc=8,use_cuda=True)
G = generator(nc=8,use_cuda=True)

encoder.cuda()
decoder.cuda()
D.cuda()
G.cuda()

# Caption optimizer
criterion = nn.CrossEntropyLoss()
caption_params = list(decoder.parameters()) + list(encoder.parameters())
caption_optimizer = torch.optim.Adam(caption_params, lr=0.001)

encoder.train()
decoder.train()

# GAN optimizer
dg_criterion = nn.BCELoss()  # 是单目标二分类交叉熵函数
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.001)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.001)
z_dimension = 32

tq_gen = tqdm(enumerate(mt_gen))
log_file = open(os.path.join(savedir, "log.txt"), "w")
cap_loss = 0.
#caption_start = 4000
caption_start = 0

for i, (mol_batch, caption, lengths) in tq_gen:
    num_img = mol_batch.size(0)
    real_data = Variable(mol_batch[:, :]).cuda()
    real_label = Variable(torch.ones(num_img)).cuda()
    fake_label = Variable(torch.zeros(num_img)).cuda()
    #print('fake label', fake_label.shape)
    ########判别器训练train#######
    real_out = D(real_data.float())  # 将真实图片放入判别器中
    
    d_loss_real = dg_criterion(real_out.view(-1), real_label)  # 得到真实图片的loss
    real_scores = real_out
    
    z = Variable(torch.randn(num_img, 128, 12, 12, 12)).cuda()

    fake_data = G(z.detach())
    fake_out = D(fake_data)
    #print(z.shape, real_data.shape, fake_data.shape, fake_out.view(-1).shape, fake_label.shape)
    d_loss_fake = dg_criterion(fake_out.view(-1), fake_label)
    fake_scores = fake_out
    
    d_loss = d_loss_real + d_loss_fake  # 损失包括判真损失和判假损失
    d_optimizer.zero_grad()  # 在反向传播之前，先将梯度归0
    d_loss.backward()  # 将误差反向传播
    d_optimizer.step()
    
    #==================训练生成器========
    z = Variable(torch.randn(num_img, 128, 12, 12, 12)).cuda()
    fake_data = G(z)
    output = D(fake_data)
    #print(output.view(-1).shape, real_label.shape)
    g_loss = dg_criterion(output.view(-1), real_label)
    g_optimizer.zero_grad()
    g_loss.backward()
    g_optimizer.step()
	
    recon_batch = G(z.detach())
    if i >= caption_start:  # Start by autoencoder optimization
        captions = Variable(caption.cuda())
        targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
        
        decoder.zero_grad()
        encoder.zero_grad()
        features = encoder(recon_batch)
        outputs = decoder(features, captions, lengths)
        cap_loss = criterion(outputs, targets)
        cap_loss.backward()
        caption_optimizer.step()
        
    break

    if (i + 1) % 5000 == 0:
        torch.save(decoder.state_dict(),
                   os.path.join(savedir,
                                'decoder-%d.pkl' % (i + 1)))
        torch.save(encoder.state_dict(),
                   os.path.join(savedir,
                                'encoder-%d.pkl' % (i + 1)))
        torch.save(G.state_dict(),
                   os.path.join(savedir,
                                'G-%d.pkl' % (i + 1)))
        torch.save(D.state_dict(),
                   os.path.join(savedir,
                                'D-%d.pkl' % (i + 1)))
        
    if (i + 1) % 100 == 0:
        result = "Step: {}, caption_loss: {:.5f}, ".format(i + 1,
                                           float(cap_loss.data.cpu().numpy()) if type(cap_loss) != float else 0.)
        log_file.write(result + "\n")
        log_file.flush()
        tq_gen.write(result)
        print('Epoch{}: d_loss={:.4f} | g_loss={:.4f} | D real score~1={:.4f} | D fake score~0={:.4f}'.format(i+1, d_loss.data.item(), g_loss.data.item(), real_scores.data.mean(), fake_scores.data.mean()))  
        
    # Reduce the LR
    if (i + 1) % 60000 == 0:
        # Command = "Reducing learning rate".format(i+1, float(loss.data.cpu().numpy()))
        log_file.write("Reducing LR\n")
        tq_gen.write("Reducing LR")
        for param_group in caption_optimizer.param_groups:
            lr = param_group["lr"] / 2.
            param_group["lr"] = lr
        
    if i == 210000:
        # We are Done!
        log_file.close()
        break

# Cleanup
del tq_gen
mt_gen.close()
multiproc.close()