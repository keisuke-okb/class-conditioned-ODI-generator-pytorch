# Class-conditioned ODI generator (PyTorch)
Simple PyTorch implementation of class-conditioned ODI generator based on [pix2pix](https://phillipi.github.io/pix2pix/)(cGAN).

## Requirements
+ numpy
+ Torch
+ Pillow

## Getting started
### Installation
+ Download sample [dataset](https://drive.google.com/file/d/1L-u-8xUg-S9KnL_7ZAJcW44pws9VHvpJ/view?usp=sharing)(Google Drive)
+ Unzip the zip file and place the `dataset` folder in the same location as the `implementation` folder

### Run
+ For GPU, run `python train.py --cuda --padding`
+ For CPU, run `python train.py --cuda --padding`




## Acknowledgments
We thank to these developers:
+ Implementation: [pix2pix-pytorch](https://github.com/mrzhu-cool/pix2pix-pytorch) by [mrzhu-cool](https://github.com/mrzhu-cool).
+ Sample dataset: [SUN360](http://people.csail.mit.edu/jxiao/SUN360/) dataset by Jianxiong Xiao, Krista A. Ehinger, Aude Oliva, and Antonio Torralba (Massachusetts Institute of Technology).
