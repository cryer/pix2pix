import torch.nn as nn
import torch.nn.functional as F
import torch
import config
import numpy as np
import torchvision.transforms as transforms
from torchvision.utils import save_image
from model import *
from data import *
from PIL import Image


cfg = config.Configs()


def test(image_path,**kwargs):
    for k, v in kwargs.items():
        setattr(cfg, k, v)
        
    device = torch.device("cuda:0" if cfg.use_gpu else "cpu")   

    generator = GeneratorUNet()
    generator = generator.to(device)
    generator.load_state_dict(torch.load('%s/generator_%d.pth' % (cfg.saved_path,cfg.test_epoch)))

    
    transforms_ = [ transforms.Resize((cfg.height, cfg.width), Image.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
    
    Tensor = torch.FloatTensor
    
    def get_image(image_path):
        img = Image.open(image_path)
        w, h = img.size
        img_A = img.crop((0, 0, w/2, h))
        img_B = img.crop((w/2, 0, w, h))
        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return {'A': img_A, 'B': img_B}

    def sample_images(image_path):
            with torch.no_grad():
                imgs = get_image(image_path)
                real_A = imgs['B'].type(Tensor).to(device)
                real_B = imgs['A'].type(Tensor).to(device)
                fake_B = generator(real_A)
                img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2)
                save_image(img_sample, 'test.png',normalize=True)
    sample_images(image_path)
            
if __name__ == '__main__':
    import fire
    fire.Fire()       
