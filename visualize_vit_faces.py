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
    heatmap_seg = heatmap.copy()
    heatmap =  cmap(heatmap)
    if np.max(heatmap) < 255.:
        heatmap *= 255
    
    heatmap_seg = (heatmap_seg[:,:,None].astype(float) * img).astype(np.uint8)

    fg = Image.fromarray(heatmap.astype('uint8')).convert('RGBA')
    fg_seg = Image.fromarray(heatmap_seg.astype('uint8')).convert('RGBA')
    img = img.convert('RGBA')
    outIm = Image.blend(img,fg,alpha=0.5)
    outImg_seg = Image.blend(img,fg_seg,alpha=1.0)
    return outIm, outImg_seg

def compute_heat_map(anchor_center, fb, anchor, fb_center, N, level):
    R = level**2
    size = (128, 128)
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
    image = Image.new('RGB', (4 * size, size))
    q_img_heatmap, q_img_seg = combine_img_heatmap(query_img, u.cpu().detach().numpy())
    right_img_heatmap, right_img_seg = combine_img_heatmap(right_img, v.cpu().detach().numpy())
    image.paste(query_img, ((0, 0)))
    # image.paste(q_img_heatmap, ((1*size, 0)))
    # image.paste(right_img_heatmap, ((2*size, 0)))
    image.paste(q_img_seg, ((1*size, 0)))
    image.paste(right_img_seg, ((2*size, 0)))
    image.paste(right_img, ((3*size, 0)))
    return image

num_pairs = 50
# input_file = '/home/hai/workspace/DeepFace_EMD/transformer_pairs_lfw128_glass_vs_lfw128_{}.txt'.format(num_pairs)
# input_file = '/home/hai/workspace/DeepFace_EMD/transformer_pairs_lfw128_masked_vs_lfw128_{}.txt'.format(num_pairs)
input_file = '/home/hai/workspace/DeepFace_EMD/transformer_pairs_lfw128_vs_lfw128_{}.txt'.format(num_pairs) 
caption = 'norm' 

data_root = '/home/hai/workspace/lfw-align-128' #InsightFace-v2/data/lfw_crop_112
# if caption == 'masked':
#     depth_vit = 8
#     heads = 1
#     out_dim =  512 #512 
#     ood_data_root = '/home/hai/workspace/lfw-align-128-masked/'# '/home/hai/workspace/lfw-align-128-glass/' # ## 
# else:
depth_vit = 1
heads = 4
out_dim =  1024
ood_data_root = '/home/hai/workspace/lfw-align-128-glass/'

GPU_ID = [0]
device = torch.device('cuda:%d' % GPU_ID[0])
torch.backends.cudnn.benchmark = True
NUM_CLASS = 10575 #93431 # #  #    #
channels = 1
size = 128 #112 
isHybrid = False
use_scale = True
grayscale = True

use_face_loss = True
use_cls = False
HEAD_NAME = 'ArcFace'

if depth_vit == 1 and heads == 4:
    model_path = 'results/ViT-face_webface_2m_mlp_ArcFace_depth_1_head_4_1024fc_lr1e5_LFW/best.pth'
    out_dim = 1024
elif depth_vit == 1 and heads == 2:
    model_path = 'results/ViT-face_webface_arcface_resnet18_gray_depth_1_head_2_lr_1e-5_fc1024_dropout_0_LFW/best.pth'
    out_dim = 1024
else:
    model_path  = 'results/ViT-P8S8_2-image_webface_2m_arcface_resnet18_s1_depth_{}_head_{}_LFW_lr1e5/best.pth'.format(depth_vit, heads) 

model = ViT_face_model(loss_type='ArcFace',
                            GPU_ID=['0'],
                            num_class=NUM_CLASS,
                            use_cls=use_cls,
                            use_face_loss=use_face_loss,
                            no_face_model=False,
                            image_size=size,
                            patch_size=8,
                            ac_patch_size=12,
                            pad = 4,
                            dim=512,
                            depth=depth_vit,
                            heads=heads,
                            mlp_dim=2048,
                            dropout=0.0,
                            emb_dropout=0.1,
                            out_dim=out_dim,
                            singleMLP=False,
                            remove_sep=False)
facemodel = resnet_face18(False, use_reduce_pool=False, grayscale=True)
model.face_model = facemodel

# model_path = 'results/ViT-face_msceleb_arcface_resnet18_gray_depth_1_head_2_lr_1e-5_fc512_dropout_0_LFW/best.pth'
# 
print('model path: {}'.format(model_path))
model.load_state_dict(torch.load(model_path))
model = model.to(device)
model.eval()

MULTI_GPU = False
level = 8

isMask = False



# img1_path = os.path.join(ood_data_root, 'Abel_Pacheco/Abel_Pacheco_0001.jpg')
# img1_path = os.path.join(data_root, 'Abel_Pacheco/Abel_Pacheco_0002.jpg')
# img2_path = os.path.join(data_root, 'Abel_Pacheco/Abel_Pacheco_0001.jpg')

with open(input_file, 'r') as ifd:
        for line in ifd:
            line = line.strip()
            parts = line.split(',')
            img1_path = parts[0].strip()
            img2_path = parts[1].strip()
            person1 = img1_path.split('/')[-2]
            person2 = img2_path.split('/')[-2]
            label = parts[-1]

            name_1 = img1_path.split('/')[-1].split('.')[0]
            name_2 = img2_path.split('/')[-1].split('.')[0]

            img1 = process_img(img1_path, size=size, grayscale=grayscale, use_scale=use_scale)
            img2 = process_img(img2_path, size=size, grayscale=grayscale, use_scale=use_scale)
            query_img = Image.open(img1_path).convert('RGB').resize((size,size))
            right_img = Image.open(img2_path).convert('RGB').resize((size,size))

            inputs = torch.cat((img1.unsqueeze(0), img2.unsqueeze(0)), 0)
            f1, f2, spatials_1, spatials_2  = model(inputs.to(device), heatmap=True)

            # embs, spatials = outputs
            f1 = torch.nn.functional.normalize(f1, p=2, dim=1)
            f2 = torch.nn.functional.normalize(f2, p=2, dim=1)
            spatials_1 = torch.nn.functional.normalize(spatials_1, p=2, dim=2)
            spatials_2 = torch.nn.functional.normalize(spatials_2, p=2, dim=2)
            u, v = compute_heat_map(f1[0], spatials_2[0], spatials_1[0], f2[0], 1, level)
            image = create_image(query_img, right_img, u, v, size)
            if person1 == person2:
                imgname = 'user_study/{}/pos/ViT_E_{}_and_{}_{}.jpg'.format(caption, person1, person2, caption)
            else:
                imgname = 'user_study/{}/neg/ViT_E_{}_and_{}_{}.jpg'.format(caption, person1, person2, caption)
            print('img: {}'.format(imgname))
            image.save(imgname)
print('done')