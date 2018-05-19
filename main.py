import torch.nn as nn
import torch.nn.functional as F
import torch
import config
import numpy as np
import math
import itertools
import sys
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from model import *
from data import *


cfg = config.Configs()

def run(**kwargs):
    for k, v in kwargs.items():
        setattr(cfg, k, v)

    device = torch.device("cuda:0" if cfg.use_gpu else "cpu")

    criterion_GAN = torch.nn.MSELoss()
    criterion_pixelwise = torch.nn.L1Loss()

    patch = (1, cfg.height//2**4, cfg.width//2**4)


    generator = GeneratorUNet()
    discriminator = Discriminator()

    generator = generator.to(device)
    discriminator = discriminator.to(device)
    criterion_GAN.to(device)
    criterion_pixelwise.to(device)


    if cfg.start_epoch != 0:
        # Load pretrained models
        generator.load_state_dict(torch.load('%s/generator_%d.pth' % (cfg.saved_path,cfg.start_epoch)))
        discriminator.load_state_dict(torch.load('%s/discriminator_%d.pth' % (cfg.saved_path,cfg.start_epoch)))
    else:
        generator.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2))

    transforms_ = [ transforms.Resize((cfg.height, cfg.width), Image.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]

    dataloader = DataLoader(ImageDataset("data/%s" % cfg.dataset_name, transforms_=transforms_),
                            batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.threads)

    val_dataloader = DataLoader(ImageDataset("data/%s" % cfg.dataset_name, transforms_=transforms_, mode='val'),
                                batch_size=10, shuffle=True, num_workers=1)


    Tensor = torch.FloatTensor

    def sample_images(batches):
        with torch.no_grad():
            imgs = next(iter(val_dataloader))
            real_A = imgs['B'].type(Tensor).to(device)
            real_B = imgs['A'].type(Tensor).to(device)
            fake_B = generator(real_A)
            img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2)
            save_image(img_sample, '%s/%s.png' % (cfg.sample_path,batches), nrow=5, normalize=True)


    for epoch in range(cfg.start_epoch, cfg.epochs):
        for i, batch in enumerate(dataloader):

            # Model inputs
            real_A = batch['B'].type(Tensor).to(device)
            real_B = batch['A'].type(Tensor).to(device)

            # Adversarial ground truths
            valid = Tensor(np.ones((real_A.size(0), *patch))).to(device)
            fake = Tensor(np.zeros((real_A.size(0), *patch))).to(device)

            #  Train Generators

            optimizer_G.zero_grad()

            # GAN loss
            fake_B = generator(real_A)
            pred_fake = discriminator(fake_B, real_A)
            loss_GAN = criterion_GAN(pred_fake, valid)
            # Pixel-wise loss
            loss_pixel = criterion_pixelwise(fake_B, real_B)

            # Total loss
            loss_G = loss_GAN + cfg.Lambda * loss_pixel

            loss_G.backward()

            optimizer_G.step()

            #  Train Discriminator

            optimizer_D.zero_grad()

            # Real loss
            pred_real = discriminator(real_B, real_A)
            loss_real = criterion_GAN(pred_real, valid)

            # Fake loss
            pred_fake = discriminator(fake_B.detach(), real_A)
            loss_fake = criterion_GAN(pred_fake, fake)

            # Total loss
            loss_D = 0.5 * (loss_real + loss_fake)

            loss_D.backward()
            optimizer_D.step()

            # log

            batches = epoch * len(dataloader) + i
            sys.stdout.write("\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f]" %
                                                            (epoch, cfg.epochs,
                                                            i, len(dataloader),
                                                            loss_D.item(), loss_G.item(),
                                                            loss_pixel.item(), loss_GAN.item()
                                                            ))

            if batches % cfg.n_sample == 0:
                sample_images(batches)


        if cfg.n_checkpoint != -1 and (epoch+1) % cfg.n_checkpoint == 0:
            torch.save(generator.state_dict(), '%s/generator_%d.pth' % (cfg.saved_path, epoch))
            torch.save(discriminator.state_dict(), '%s/discriminator_%d.pth' % (cfg.saved_path, epoch))

if __name__ == '__main__':
    import fire
    fire.Fire()
