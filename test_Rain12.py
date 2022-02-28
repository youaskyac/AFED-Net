import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import *
from skimage.measure.simple_metrics import compare_psnr
from skimage.measure._structural_similarity import compare_ssim
from matplotlib import pyplot as plt
from model_ab4_5blc import *

import time

parser = argparse.ArgumentParser(description="Test_Rain12")
parser.add_argument("--logdir", type=str, default="logs/Rain100H_5blc", help='path of log files')
parser.add_argument("--data_path", type=str, default="./datasets/test/Rain12", help='path to test data')
parser.add_argument("--save_path", type=str, default="results/Rain12_5blc", help='path to save results')
parser.add_argument("--use_GPU", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')

opt = parser.parse_args()

if opt.use_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

# opt.logdir = opt.logdir + "_inter%d" % opt.inter_iter + "_intra%d" % opt.intra_iter
# opt.save_path = os.path.join(opt.data_path, opt.save_path)
# opt.save_path = opt.save_path + "_inter%d" % opt.inter_iter + "_intra%d" % opt.intra_iter

if not os.path.exists(opt.save_path):
    os.mkdir(opt.save_path)


def normalize(data):
    return data / 255.


def main():
    # Build model
    print('Loading model ...\n')

    model = NET(input_channel=32)

    if opt.use_GPU:
        model = model.cuda()

    if opt.use_GPU:
        model = model.cuda()
    model.load_state_dict(torch.load(os.path.join(opt.logdir, 'net_latest.pth')))  # 导入最终的训练模型

    model.eval()
    # load data info
    print('Loading data info ...\n')
    files_source = glob.glob(os.path.join(opt.data_path, 'rainy/*.png'))
    target_source = glob.glob(os.path.join(opt.data_path, 'norain/*.png'))
    print(files_source)
    print(target_source)
    # print('--------------------files_source------------------')
    # print(files_source)
    # print('--------------------target------------------')
    # print(target_source)
    files_source.sort()
    # process data
    time_test = 0
    sum_psnr = 0
    sum_ssim = 0
    count = 0
    for f, t in zip(files_source, target_source):
        count = count+1
        img_name = os.path.basename(f)
        target_name =  os.path.basename(t)
        # image
        Img = cv2.imread(f)
        target = cv2.imread(t)
        target = cv2.cvtColor(target,cv2.COLOR_BGR2RGB)
        # print(Img.shape)
        # print(target.shape)
        b, g, r = cv2.split(Img)
        Img = cv2.merge([r, g, b])
        # Img = cv2.resize(Img, (int(1024), int(1024)), interpolation=cv2.INTER_CUBIC)
        Img = normalize(np.float32(Img))
        Img = np.expand_dims(Img.transpose(2, 0, 1), 0)
        ISource = torch.Tensor(Img)

        if opt.use_GPU:
            ISource = Variable(ISource.cuda())
        else:
            ISource = Variable(ISource)

        with torch.no_grad():  # this can save much memory
            if opt.use_GPU:
                torch.cuda.synchronize()
            start_time = time.time()
            out = model(ISource)
            out = torch.clamp(out, 0., 1.)
            # print('---------------------')
            # print(out.shape)
            # target =target.transpose(2,0,1)
            # target = target[np.newaxis,:,:,:]
            # print(target.shape)
            out_copy = out.clone()
            out_copy = out_copy.data.cpu().numpy().astype(np.float32)
            out_copy = out_copy[0,:,:,:]
            out_copy = out_copy.transpose(1,2,0)
            plt.subplot(121)
            plt.imshow(target)
            plt.subplot(122)
            plt.imshow(out_copy)
            plt.show()
            # print(out_copy.shape)
            # print(out_copy)
            # print(target)
            target = np.float32(target/255)
            Psnr = compare_psnr(out_copy, target, 1.)
            Ssim = compare_ssim(out_copy, target, win_size=None, gradient=False,
                                data_range=None, multichannel=True)
            sum_psnr += Psnr
            sum_ssim +=Ssim
            print(Psnr)
            print(Ssim)

            if opt.use_GPU:
                torch.cuda.synchronize()
            end_time = time.time()
            dur_time = end_time - start_time
            print(img_name)
            print(target_name)
            print(dur_time)
            time_test += dur_time

        if opt.use_GPU:
            save_out = np.uint8(255 * out.data.cpu().numpy().squeeze())
        else:
            save_out = np.uint8(255 * out.data.numpy().squeeze())

        save_out = save_out.transpose(1, 2, 0)
        b, g, r = cv2.split(save_out)

        save_out = cv2.merge([r, g, b])

        save_path = opt.save_path
        cv2.imwrite(os.path.join(save_path, img_name), save_out)
    avg_psnr = sum_psnr/count
    avg_ssim =sum_ssim/count
    print(time_test / 100)
    print("Avg_Psnr=")
    print(avg_psnr)
    print("Avg_ssim=")
    print(avg_ssim)


if __name__ == "__main__":
    main()
