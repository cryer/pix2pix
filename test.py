import torch
import config
import numpy as np
import torchvision.transforms as transforms
from torchvision.utils import save_image
from model import *
from data import *
from PIL import Image
from torch.utils.data import DataLoader

cfg = config.Configs()


def test(**kwargs):
    for k, v in kwargs.items():
        setattr(cfg, k, v)
        
    device = torch.device("cuda:0" if cfg.use_gpu else "cpu")   

    generator = GeneratorUNet()
    generator = generator.to(device)
    generator.load_state_dict(torch.load('%s/generator_%d.pth' % (cfg.saved_path,cfg.test_epoch)))

    transforms_ = [ transforms.Resize((cfg.height, cfg.width), Image.BICUBIC),
                   transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
    val_dataloader = DataLoader(ImageDataset("data/%s" % cfg.dataset_name, transforms_=transforms_, mode='val'),
                                batch_size=10, shuffle=True, num_workers=1)
    
    Tensor = torch.FloatTensor

    with torch.no_grad():
        imgs = next(iter(val_dataloader))
        real_A = imgs['B'].type(Tensor).to(device)
        real_B = imgs['A'].type(Tensor).to(device)
        fake_B = generator(real_A)
        img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2)
        save_image(img_sample, 'test.png',nrow=5,normalize=True)

            
if __name__ == '__main__':
    import fire
    fire.Fire()       
      
