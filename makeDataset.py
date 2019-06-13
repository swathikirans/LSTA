import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import random

class makeDataset(Dataset):
    def __init__(self, dataset, labels, numFrames, spatial_transform=None, seqLen=30,
                 train=True, mulSeg=False, numSeg=1, fmt='.jpg', mode='train'):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.spatial_transform = spatial_transform
        self.train = train
        self.mulSeg = mulSeg
        self.numSeg = numSeg
        self.images = dataset
        self.labels = labels
        self.numFrames = numFrames
        self.seqLen = seqLen
        self.fmt = fmt
        self.mode = mode

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        vid_name = self.images[idx]
        label = self.labels[idx]
        numFrame = self.numFrames[idx]
        inpSeq = []
        # print('Vid {} | numFrames = {}'.format(vid_name, numFrame))
        self.spatial_transform.randomize_parameters()
        for i in np.linspace(1, numFrame, self.seqLen, endpoint=True):
            fl_name = vid_name + '/' + 'image_' + str(int(np.floor(i))).zfill(5) + self.fmt
            img = Image.open(fl_name)
            inpSeq.append(self.spatial_transform(img.convert('RGB')))
        inpSeq = torch.stack(inpSeq, 0)
        if self.mode == 'eval':
            return inpSeq, label, vid_name
        else:
            return inpSeq, label
