# Single Image Deraining Using Attention Mechanism Fused Encoder-Decoder Network
## Introduction
In this paper, we propose a novel end-to-end deraining network AFED-Net. The network first extracts rich contextual information at different scales, and then introduces supervised information to optimize feature propagation; then uses the proposed spatial attention fusion mechanism to solve the semantic information inconsistency caused by different scales.  The proposed attention supervision module not only plays the effect of feature meritocracy, its supervision information generated from the original images can also subsequent networks can more effectively compensate for the missing spatial details caused by the down-sampling operation. Experimental results on several publicly available synthetic rain datasets and real rain datasets show that the proposed network model plays a good role and outperforms several existing rain removal algorithms in terms of rain removal performance.
## Prerequisites

- Python 3.6, PyTorch >= 0.4.0 
- Requirements: opencv-python, tensorboardX
- Platforms: Ubuntu 16.04, cuda-8.0 & cuDNN v-5.1 (higher versions also work well)
## Datasets

our model are evaluated on three datasets: 
Rain100H , Rain100L , Rain12 ，Rain1200 ,Rain1400   
Please download the testing datasets from [BaiduYun](https://pan.baidu.com/s/115-OBqATI9JGS3ZG0-BUsA) Access Code ：xbts   
and place the unzipped folders into `./datasets/test/`.


## Getting Started

### 1) Testing

We have placed our pre-trained models into `./logs/`. 
Run scripts to test the models:

```python
python test_Rain100H.py   # test models on Rain100H
python test_Rain100L.py   # test models on Rain100L
python test_real.py       # test models on real rainy images

