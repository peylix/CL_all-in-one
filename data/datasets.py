import os
import cv2
import torch
import random
import numpy as np
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as tfs

from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
from torchvision.transforms import functional as FF



def normalize(data):
    return data / 255.

def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw - win + 0 + 1:stride, 0:endh - win + 0 + 1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win * win, TotalPatNum], np.float32)

    for i in range(win):
        for j in range(win):
            patch = img[:, i:endw - win + i + 1:stride, j:endh - win + j + 1:stride]
            Y[:, k, :] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])



class PairedImageDataset(data.Dataset):
    """Generic paired image dataset.
    Expects two directories: input_dir (degraded) and gt_dir (ground truth).
    If filenames differ between input and gt, provide gt_name_fn to map
    input filename to gt filename (e.g. '0_rain.png' -> '0_clean.png').
    """
    def __init__(self, input_dir, gt_dir, size=240, gt_name_fn=None):
        super(PairedImageDataset, self).__init__()
        self.size = size
        self.gt_name_fn = gt_name_fn
        self.input_imgs_dir = sorted(os.listdir(input_dir))
        self.input_imgs = [os.path.join(input_dir, img) for img in self.input_imgs_dir]
        self.gt_dir = gt_dir

    def __getitem__(self, index):
        inp = Image.open(self.input_imgs[index])
        if isinstance(self.size, int):
            while inp.size[0] < self.size or inp.size[1] < self.size:
                index = random.randint(0, len(self.input_imgs) - 1)
                inp = Image.open(self.input_imgs[index])
        filename = os.path.basename(self.input_imgs[index])
        if self.gt_name_fn:
            filename = self.gt_name_fn(filename)
        gt = Image.open(os.path.join(self.gt_dir, filename))
        gt = tfs.CenterCrop(inp.size[::-1])(gt)
        if isinstance(self.size, int):
            i, j, h, w = tfs.RandomCrop.get_params(inp, output_size=(self.size, self.size))
            inp = FF.crop(inp, i, j, h, w)
            gt = FF.crop(gt, i, j, h, w)
        inp = tfs.ToTensor()(inp.convert("RGB"))
        inp = tfs.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])(inp)
        gt = tfs.ToTensor()(gt.convert("RGB"))
        return inp, gt

    def __len__(self):
        return len(self.input_imgs)


def _raindrop_gt_name(filename):
    """Map raindrop input filename to gt filename: '0_rain.png' -> '0_clean.png'"""
    return filename.replace('_rain', '_clean')

def get_trainloader(args):
    path = args.data_path

    haze_traindata = PairedImageDataset(path+'/CVPR19RainTrain/train/data', path+'/CVPR19RainTrain/train/gt', size=args.crop_size)
    rain_traindata = PairedImageDataset(path+'/raindrop_data/train/data', path+'/raindrop_data/train/gt', size=args.crop_size, gt_name_fn=_raindrop_gt_name)
    snow_traindata = PairedImageDataset(path+'/Snow100K-training/synthetic', path+'/Snow100K-training/gt', size=args.crop_size)

    haze_loadertrain = DataLoader(dataset=haze_traindata, batch_size=args.bs, shuffle=True)
    rain_loadertrain = DataLoader(dataset=rain_traindata, batch_size=args.bs, shuffle=True)
    snow_loadertrain = DataLoader(dataset=snow_traindata, batch_size=args.bs, shuffle=True)

    TASK = {'haze': haze_loadertrain, 'rain': rain_loadertrain, 'snow': snow_loadertrain}

    trainloader = []
    for task in args.task_order:
        trainloader.append(TASK[task])
    return trainloader

def get_testloader(args):
    path = args.data_path

    haze_testdata = PairedImageDataset(path+'/CVPR19RainTrain/test/data', path+'/CVPR19RainTrain/test/gt', size='whole img')
    rain_testdata = PairedImageDataset(path+'/raindrop_data/test_a/data', path+'/raindrop_data/test_a/gt', size='whole img', gt_name_fn=_raindrop_gt_name)
    snow_testdata = PairedImageDataset(path+'/Snow100K-testing/jdway/GameSSD/overlapping/test/Snow100K-M/synthetic', path+'/Snow100K-testing/jdway/GameSSD/overlapping/test/Snow100K-M/gt', size='whole img')

    haze_loadertest = DataLoader(dataset=haze_testdata, batch_size=1, shuffle=True)
    rain_loadertest = DataLoader(dataset=rain_testdata, batch_size=1, shuffle=True)
    snow_loadertest = DataLoader(dataset=snow_testdata, batch_size=1)

    TASK = {'haze': haze_loadertest, 'rain': rain_loadertest, 'snow': snow_loadertest}

    testloader = []
    for task in args.task_order:
        testloader.append(TASK[task])
    return testloader
