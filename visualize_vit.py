from scipy.misc import face
from models.vit_model_face import ViT_face_model, Hybrid_ViT
from models.model_irse_norm import norm_50
from models.resnet import resnet_face18
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import sys
from models import ViT_face
from models import ViTs_face
# from models import ViT_model_face
from util.utils import get_val_data, perform_val, process_img
from IPython import embed
import sklearn
import cv2
import numpy as np
from image_iter import FaceDataset
import torch.utils.data as data
import argparse
import os
from PIL import Image


def combine_img_heatmap(img, heatmap):
    cmap = plt.get_cmap('jet') # colormap for the heatmap
    heatmap = heatmap - np.min(heatmap)
    heatmap /= np.max(heatmap)
    heatmap =  cmap(heatmap)
    if np.max(heatmap) < 255.:
        heatmap *= 255
    
    fg = Image.fromarray(heatmap.astype('uint8')).convert('RGBA')
    img = img.convert('RGBA')
    outIm = Image.blend(img,fg,alpha=0.5)
    return outIm

def compute_heat_map(anchor_center, fb, anchor, fb_center, N, level):
    R = level**2
    size = (112, 112)
    shape = [(0, 0), (size[0], size[1])]
    # att = F.relu(torch.einsum("c,ncr->nr", anchor_center, fb)).view(N, R)
    att = F.relu(torch.einsum("c,rc->r", anchor_center, fb)).view(N, R)
    u = att / (att.sum(dim=1, keepdims=True) + 1e-7)
    att = F.relu(torch.einsum("rc,c->r", anchor, fb_center)).view(N, R)
    v = att / (att.sum(dim=1, keepdims=True) + 1e-7)

    u, v = u.view(level,level), v.view(level,level)
    u, v = u.view(1,1,level,level), v.view(1,1,level,level)
    u = F.interpolate(u,shape[1],mode='bilinear',align_corners=True).view(size[0], size[1])
    v = F.interpolate(v,shape[1],mode='bilinear',align_corners=True).view(size[0], size[1])
    return u, v

def create_image(query_img, right_img, u, v, size):
    image = Image.new('RGB', (2 * size, size))
    q_img_heatmap = combine_img_heatmap(query_img, u.cpu().detach().numpy())
    right_img_heatmap = combine_img_heatmap(right_img, v.cpu().detach().numpy())
    image.paste(q_img_heatmap, ((0, 0)))
    image.paste(right_img_heatmap, ((1*size, 0)))
    return image


GPU_ID = [0]
device = torch.device('cuda:%d' % GPU_ID[0])
torch.backends.cudnn.benchmark = True
NUM_CLASS = 93431 #    #
depth_vit = 20
heads = 8
channels = 1
size = 112 
out_dim = 512 
use_scale = False
grayscale = False
HEAD_NAME = 'ArcFace'


model = ViT_face(loss_type = HEAD_NAME,
                    GPU_ID = GPU_ID,
                    num_class = NUM_CLASS,
                    image_size=112,
                    patch_size=8,
                    dim=512,
                    depth=depth_vit,
                    heads=heads,
                    mlp_dim=2048,
                    dropout=0.1,
                    emb_dropout=0.1)

model_path  = 'results/ViT-P8S8_ms1m_cosface_s1_continue/best.pth'
print('model path: {}'.format(model_path))
model.load_state_dict(torch.load(model_path))
model = model.to(device)

MULTI_GPU = False
level = 14

caption = 'glass'
data_root = '/home/hai/workspace/lfw-align-128' #InsightFace-v2/data/lfw_crop_112
if caption == 'masked':
    ood_data_root = '/home/hai/workspace/lfw-align-128-masked/'
else:
    ood_data_root = '/home/hai/workspace/lfw-align-128-glass/'

img1_path = os.path.join(ood_data_root, 'Lisa_Raymond/Lisa_Raymond_0001.jpg')

# img1_path = os.path.join(data_root, 'Abel_Pacheco/Abel_Pacheco_0002.jpg')
img2_path = os.path.join(data_root, 'Lisa_Raymond/Lisa_Raymond_0001.jpg')

name_1 = img1_path.split('/')[-1].split('.')[0]
name_2 = img2_path.split('/')[-1].split('.')[0]
# img2_path = os.path.join(data_root, 'Abel_Pacheco/Abel_Pacheco_0002.jpg')

img1 = process_img(img1_path, size=size, grayscale=grayscale, use_scale=use_scale)
img2 = process_img(img2_path, size=size, grayscale=grayscale, use_scale=use_scale)
query_img = Image.open(img1_path).convert('RGB').resize((size,size))
right_img = Image.open(img2_path).convert('RGB').resize((size,size))

inputs = torch.cat((img1.unsqueeze(0), img2.unsqueeze(0)), 0)
outputs = model(inputs.to(device), vis=True)

embs, spatials = outputs
embs = torch.nn.functional.normalize(embs, p=2, dim=1)
spatials = torch.nn.functional.normalize(spatials, p=2, dim=2)
u, v = compute_heat_map(embs[0], spatials[1], spatials[0], embs[1], 1, level)
image = create_image(query_img, right_img, u, v, size)
imgname = 'heatmap_face_vit_{}_vs_{}_{}.jpg'.format(name_1, name_2, caption)
print('img: {}'.format(imgname))
image.save(imgname)
print('done')
