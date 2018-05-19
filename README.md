# pix2pix
a PyTorch implementation of Image-to-Image Translation with Conditional Adversarial Networks

# Paper and official code

* [Arxiv](https://arxiv.org/abs/1611.07004)
* [Code](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

# Introduction

Pix2Pix is a Image-to-Image translation project,it can do many kinds of things,just shows some of them below:

![](https://github.com/cryer/pix2pix/raw/master/images/1.png)

It's based on conditional GAN,where the condition is not a vector or what but a image.just like below:

![](https://github.com/cryer/pix2pix/raw/master/images/2.png)

## Generator

Paper compares two different generators,Enocder-Decoder and U-Net. Result shows U-Net do a better job,maybe because U-Net has
some skip connections,this leads to understand better about low-level features.

![](https://github.com/cryer/pix2pix/raw/master/images/3.png)

## Discriminator

Paper uses patchGAN as discriminator,which means we dont judge the whole image pair,but judge some patchs of images,then do an average.
This speeds up training phase,and can process different size of images.

# Datasets

Team also releases some nice datasets,you can download freely.[Download Link](https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/)

I will use cityscapes dataset. You can download your favorite datasets and put into data subdirectory.

# Requirements

* pytorch 0.4.0
* torchvision
* fire

I used pytorch 0.4.0 to build this project,so you need to update your pytorch version.And pytorch 0.4.0 has a total change from lower
version,so if you are using pytorch 0.1-0.3,here is a official [Migration Guide](https://pytorch.org/2018/04/22/0_4_0-migration-guide.html).

# Get Started

If you satisfy all requirements,just run:
```
git clone https://github.com/cryer/pix2pix.git
cd pix2pix
python main.py run --dataset_name=cityscapes
```
Default we use GPU and train 300 epochs,if you use other datasets,change dataset_name to yours.
All configs are:
```
    --start_epoch = 0  # start epoch to train
    --test_epoch = 0   # checkpoints to use for test
    --epochs = 300     # training epochs
    --dataset_name = "cityscapes"   
    --saved_path = "checkpoints"
    --sample_path = "sample"
    --use_gpu = True
    --batch_size = 1
    --decay_epoch = 100  # learning_rate decay,but i did not use
    --learning_rate = 0.0002
    --threads = 8   # threads to use for loading datasets
    --width = 256
    --height = 256
    --channels = 3
    --n_sample = 1000     # sample images interval
    --n_checkpoint = 50   # save checkpoints interval
    --beta1 = 0.5
    --beta2 = 0.999
    --Lambda = 100  # balance L1 loss and cGAN loss
```
Change them if you need.

# Results

First row are input,second row are generated,third row are ground truth.

Sort by time.

![](https://github.com/cryer/pix2pix/raw/master/images/10000.png)

![](https://github.com/cryer/pix2pix/raw/master/images/128500.png)

![](https://github.com/cryer/pix2pix/raw/master/images/542500.png)

![](https://github.com/cryer/pix2pix/raw/master/images/594500.png)

# Conclusion

I have trained 200 epochs on a single TiTanX for about 15 hours.We can see generated images tend to be clearer through time.However,it 
seems we need to train more epochs.

# Test

To test with trained checkpoints,run:
```
python test.py test --test_epoch=299
```
Checkpoints file ``` generator_299.pth```is trained with 300 epochs,about 207 MB.
