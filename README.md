# Class-conditioned ODI generator (PyTorch)

Simple PyTorch implementation of class-conditioned ODI generator based on [pix2pix](https://phillipi.github.io/pix2pix/)(cGAN).

**Paper: [Omni-Directional Image Generation from Single Snapshot Image]() by Keisuke Okubo and Takao Yamanaka.**


![concept](https://user-images.githubusercontent.com/70097451/91708069-bf5a4f80-ebbb-11ea-9a62-45784a3d6af6.jpg)

## Outline of ODI generation system with class-conditioned generator
![method_class_conditioned_v3](https://user-images.githubusercontent.com/70097451/91708633-8d95b880-ebbc-11ea-9545-724c96fd73c4.jpg)



## Requirements
+ numpy
+ Torch
+ Pillow

## Getting started
### Installation
+ Clone or download this repository.
+ Download [sample dataset](https://drive.google.com/file/d/1L-u-8xUg-S9KnL_7ZAJcW44pws9VHvpJ/view?usp=sharing)(Google Drive).
+ Unzip the zip file and place the `dataset` folder in the same location as the `implementation` folder.

```
class-conditioned-ODI-generator-pytorch/
  +- implementation/
  |    +- train.py
  |    +- networks.py
  |    +- ...
  +- dataset/
       +- train/
       |    +- base/
       |    +- label/
       +- test/
            +- base/
            +- label/
```

+ Move to the implementation directory: `cd implementation`

### Run
+ For GPU, run the following command 
```
python train.py --cuda
```
+ For CPU, run the following command
```
python train.py
```
### Use padding method
+ Run with this argument: `--padding`

### Optional arguments
\# of epoch: `--niter <int>` + `--niter_decay <int>`
+ `--niter <int>`: # of iter at starting learning rate. Default: 100
+ `--niter_decay <int>`: # of iter to linearly decay learning rate to zero. Default: 0

+ `--g_ch <int>`: Generator channels in first conv layer. Default: 128
+ `--d_ch <int>`: Discriminator channels in first conv layer. Default: 128
+ `--save_interval <int>`: Interval epoch of network weight saving. Default: 10
+ `--graph_save_while_training`: If save current loss graph while training. Default: False



## Acknowledgments
We thank to these developers:
+ Implementation: [pix2pix-pytorch](https://github.com/mrzhu-cool/pix2pix-pytorch) by [mrzhu-cool](https://github.com/mrzhu-cool).
+ Sample dataset: Using part of the [SUN360](http://people.csail.mit.edu/jxiao/SUN360/) dataset by Jianxiong Xiao, Krista A. Ehinger, Aude Oliva, and Antonio Torralba (Massachusetts Institute of Technology).
