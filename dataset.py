import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np
import cv2
import os
import torch

import mxnet as mx
from mxnet import ndarray as nd
from mxnet import io
from mxnet import recordio
import logging
import numbers
import random
import os
import cv2

def process_img(img_path, size=112, grayscale=False, use_scale=True):
    if grayscale:
        img = cv2.imread(img_path, 0)
        img = cv2.resize(img, (size,size))
        img = img.reshape((size,size,1))
    else:
        img = cv2.imread(img_path)
        img = cv2.resize(img, (size,size))
        img = img.reshape((size,size,3))
        if use_scale == False:
            img = img[:,:,::-1]

    img = img.transpose((2, 0, 1))
    img = img.astype(np.float32, copy=False)
    if use_scale:
        img -= 127.5
        img /= 127.5
    img = torch.from_numpy(img).float()
    return img

class FaceImgDataset(data.Dataset):
    def __init__(self, imgfile, size=128, grayscale=False, use_scale=True):
        self.images = []
        self.labels = []
        self.size = size
        self.grayscale = grayscale
        self.use_scale = use_scale
        with open(imgfile, 'r') as ifd:
            for line in ifd:
                line = line.strip()
                splits = line.split(',')
                imgpath = splits[0]
                label = splits[-1]
                self.images.append(imgpath)
                self.labels.append(label)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img = process_img(self.images[index], size=self.size, grayscale=self.grayscale, use_scale=self.use_scale)
        label = np.zeros((1,1), np.float32)
        label[0,0] = self.labels[index]
        return img, label

class FaceDataset(data.Dataset):
    def __init__(self, path_imgrec, rand_mirror):
        self.rand_mirror = rand_mirror
        assert path_imgrec
        if path_imgrec:
            logging.info('loading recordio %s...',
                         path_imgrec)
            path_imgidx = path_imgrec[0:-4] + ".idx"
            print(path_imgrec, path_imgidx)
            self.imgrec = recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
            s = self.imgrec.read_idx(0)
            header, _ = recordio.unpack(s)
            if header.flag > 0:
                print('header0 label', header.label)
                self.header0 = (int(header.label[0]), int(header.label[1]))
                # assert(header.flag==1)
                # self.imgidx = range(1, int(header.label[0]))
                self.imgidx = []
                self.id2range = {}
                self.seq_identity = range(int(header.label[0]), int(header.label[1]))
                for identity in self.seq_identity:
                    s = self.imgrec.read_idx(identity)
                    header, _ = recordio.unpack(s)
                    a, b = int(header.label[0]), int(header.label[1])
                    count = b - a
                    self.id2range[identity] = (a, b)
                    self.imgidx += range(a, b)
                print('id2range', len(self.id2range))
            else:
                self.imgidx = list(self.imgrec.keys)
            self.seq = self.imgidx

    def __getitem__(self, index):
        idx = self.seq[index]
        s = self.imgrec.read_idx(idx)
        header, s = recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        _data = mx.image.imdecode(s)
        if self.rand_mirror:
            _rd = random.randint(0, 1)
            if _rd == 1:
                _data = mx.ndarray.flip(data=_data, axis=1)

        _data = nd.transpose(_data, axes=(2, 0, 1))
        _data = _data.asnumpy()
        img = torch.from_numpy(_data)

        return img, label

    def __len__(self):
        return len

class FacePairDataset(data.Dataset):
    def __init__(self, root, pair_file, size=112, use_scale=True, grayscale=True):
        self.pairs = set()
        self.root = root
        self.use_scale = use_scale
        self.gray = grayscale
        self.size = size
        with open(pair_file, 'r') as ifd:
            for line in ifd:
                line = line.strip()
                splits = line.split(' ')
                img_name_1, img_name_2, label = splits
                pair = (img_name_1, img_name_2, float(label))
                self.pairs.add(pair)
        self.pairs = list(self.pairs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        pair = self.pairs[index]
        img_name_1, img_name_2, label = pair
        img1 = process_img(os.path.join(self.root, img_name_1), size=self.size, use_scale=self.use_scale, grayscale=self.gray)
        img2 = process_img(os.path.join(self.root, img_name_2), size=self.size, use_scale=self.use_scale, grayscale=self.gray)
        labels = np.zeros((1,1), np.float32)
        labels[0,0] = label
        return img1, img2, label

class FacePairLabelDataset(data.Dataset):
    def __init__(self, root, pair_file, size=112, grayscale=False, use_scale=True):
        self.pairs = set()
        self.root = root
        self.size = size
        self.gray = grayscale
        self.use_scale = use_scale
        with open(pair_file, 'r') as ifd:
            for line in ifd:
                line = line.strip()
                splits = line.split(' ')
                img_name_1, img_name_2, score, l1, l2 = splits
                pair = (img_name_1, img_name_2, float(score), int(l1), int(l2))
                self.pairs.add(pair)
        self.pairs = list(self.pairs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        pair = self.pairs[index]
        img_name_1, img_name_2, score, l1, l2 = pair
        img1 = process_img(os.path.join(self.root, img_name_1), size=self.size, grayscale=self.gray, use_scale=self.use_scale)
        img2 = process_img(os.path.join(self.root, img_name_2), size=self.size, grayscale=self.gray, use_scale=self.use_scale)
        score = np.zeros((1,1), np.float32)
        score[0, 0] = score
        
        label1 = np.zeros((1,1), np.float32)
        label1[0,0] = l1

        label2 = np.zeros((1,1), np.float32)
        label2[0,0] = l2
        return img1, img2, score, label1, label2