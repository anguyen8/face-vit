import wandb
import os, argparse, sklearn
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import torch.nn.functional as F
import matplotlib.pyplot as plt

from util.utils import process_img
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from PIL import Image
from models import ViT_face
from models import ViTs_face
from models.vit_model_face import ViT_face_model
from models.model_irse_norm import norm_50
from models.model_irse import IR_50
from models.resnet import resnet_face18
from IPython import embed
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer

def get_patch_location(index, size, level=8):
    patch_size = int(size / level)
    row=int(index/level)
    col=int(index%level)

    row_location = (row * patch_size, (row + 1) * patch_size) 
    col_location = (col * patch_size, (col + 1) * patch_size)
    return row_location, col_location, row, col

def compute_heat_map(u,v, level):
    R = level**2
    size = (128, 128)
    shape = [(0, 0), (size[0], size[1])]
    # u = u / (u.sum(dim=1, keepdims=True) + 1e-7)
    # v = v / (v.sum(dim=1, keepdims=True) + 1e-7)

    u, v = u.view(level,level), v.view(level,level)
    u, v = u.view(1,1,level,level), v.view(1,1,level,level)
    u = F.interpolate(u,shape[1],mode='bilinear',align_corners=True).view(size[0], size[1])
    v = F.interpolate(v,shape[1],mode='bilinear',align_corners=True).view(size[0], size[1])
    return u, v

def select_uv(u, v):
    r, c = u.shape[0], u.shape[1]
    s = int(np.sqrt(r))
    u = torch.diagonal(u, 0).view(s, s)
    v = torch.diagonal(v, 0).view(s, s)   
    u = torch.nn.functional.normalize(u, p=2, dim=1)
    v = torch.nn.functional.normalize(v, p=2, dim=1) 
    return u, v
    # for i in range(r):
    #     row = max(u[])

def removeCLSandSep(x):
    h, w = x.size()
    mid = int(h/2)
    # remove rows
    x = x[1:, :]
    x1, x2 = x[0:mid-1, :], x[mid:,:]
    x = torch.cat((x1,x2), dim=0)

    # remove cols
    x = x[:, 1:]
    x1, x2 = x[:, 0:mid-1], x[:,mid:]
    x = torch.cat((x1,x2), dim=-1)
    # x = x[0:mid-1, mid-1:w-2]
    # dia = 1.0 - torch.diagonal(x, 0)
    # for i in range(mid - 1):
    #     x[i,i] = dia[i]
    u = x[0:mid-1, mid-1:w-2]
    v = x[mid-1:w-2, 0:mid-1]
    u, v = select_uv(u, v)
    # x = torch.nn.functional.normalize(x, p=2, dim=1)
    return u, v

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

def create_image(query_img, right_img, u, v, size):
    image = Image.new('RGB', (2 * size, size))
    q_img_heatmap = combine_img_heatmap(query_img, u.cpu().detach().numpy())
    right_img_heatmap = combine_img_heatmap(right_img, v.cpu().detach().numpy())
    image.paste(q_img_heatmap, ((0, 0)))
    image.paste(right_img_heatmap, ((1*size, 0)))
    return image

if __name__ == '__main__':
    # model_path = 'results/ViT-face_webface_arcface_resnet18_gray_depth_1_3e4/Backbone_VIT_face_Epoch_3_Batch_500_Time_2022-02-26-19-24_checkpoint.pth'
    # face_transformer_path = 'results/ViT-face_webface_2m_depth_2_resnet18_gray_arcface_1e5/Backbone_VIT_face_Epoch_47_Batch_2000_Time_2022-03-04-00-11_checkpoint.pth'
    face_transformer_path = 'results/ViT-face_webface_arcface_resnet18_gray_depth_1_head_2_lr_1e-5_fc1024_dropout_0_LFW/best.pth'
    # face_transformer_path = 'results/ViT-P8S8_2-image_webface_2m_arcface_s1_depth_1_head_2_remove_pos/best.pth'
    GPU_ID = [0]
    device = torch.device('cuda:%d' % GPU_ID[0])
    torch.backends.cudnn.benchmark = True
    NUM_CLASS = 10575
    depth_vit = 1
    heads = 2
    size = 128
    self_att = True
    remove_pos = False
    dim = 1024
    transformRaw = transforms.Compose([
                    transforms.Resize([size, size]),
                    transforms.ToTensor()])
    model = ViT_face_model(loss_type='ArcFace',
                         GPU_ID=['0'],
                         num_class=NUM_CLASS,
                         use_cls=False,
                         use_face_loss=True,
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
                         out_dim=dim,
                         singleMLP=False,
                         remove_sep=False,
                         remove_pos=remove_pos)
    
    facemodel = resnet_face18(False, use_reduce_pool=False, grayscale=True)
    model.face_model = facemodel
    model.load_state_dict(torch.load(face_transformer_path))
    model = model.cuda()
    model.eval()
    layer_id = 0
    head_id = 0
    caption = 'glass'  #'diff' #'mask' #'identical' ##
    data_root = '/home/hai/workspace/lfw-align-128' #InsightFace-v2/data/lfw_crop_112
    ood_data_root = '/home/hai/workspace/lfw-align-128-glass/' #'/home/hai/workspace/lfw-align-128-masked/' #  #
    
    # img1_path = os.path.join(ood_data_root, 'Thierry_Falise/Thierry_Falise_0002.jpg')
    # img1_path = os.path.join(ood_data_root, 'Werner_Schlager/Werner_Schlager_0001.jpg')
    # img2_path = os.path.join(data_root, 'Werner_Schlager/Werner_Schlager_0001.jpg')
    

    # img1_path = os.path.join(mask_data_root, 'Abel_Pacheco/Abel_Pacheco_0001.jpg')
    img2_path = os.path.join(ood_data_root, 'Abel_Pacheco/Abel_Pacheco_0001.jpg')
    
    
    # img1_path = os.path.join(data_root, 'Charles_Kartman/Charles_Kartman_0001.jpg')
    img1_path = os.path.join(data_root, 'Abel_Pacheco/Abel_Pacheco_0001.jpg') 
    # img2_path = os.path.join(data_root, 'Charles_Kartman/Charles_Kartman_0001.jpg')
    # img2_path = os.path.join(data_root, 'Debra_Shank/Debra_Shank_0001.jpg')
    
    name_1 = img1_path.split('/')[-1].split('.')[0]
    name_2 = img2_path.split('/')[-1].split('.')[0]

    person1 = img1_path.split('/')[-2]
    person2 = img2_path.split('/')[-2]
    img1 = process_img(img1_path, grayscale=True, size=size)
    img2 = process_img(img2_path, grayscale=True, size=size)
    inputs = torch.cat((img1.unsqueeze(0), img2.unsqueeze(0)), 0)
    f1, f2, attns = model(inputs.cuda(), fea=True, vis=True)
    _, heads, dim, _ = attns[0].size()
    dim -= 2
    num = int(dim / 2)
    level = int(np.sqrt(num))

    query_img = Image.open(img1_path).convert('RGB').resize((size,size))
    right_img = Image.open(img2_path).convert('RGB').resize((size,size))

    for d in range(depth_vit):
        for hid in range(heads):
            print('processing layer {}, head id: {} ... '.format(d, hid))
            u, v =  removeCLSandSep(attns[d][0,hid, :, :])
            u, v = compute_heat_map(u, v, 8)
            image = create_image(query_img, right_img, u, v, size)
            # cross_img, cross_mat = draw_img_with_flow(flow_cross, query_img, right_img, transformRaw, size=size)

            
            heatmap_imgname = 'heatmaps/heatmap_face_{}_{}_vs_{}_depth_{}_did_{}_head_{}.jpg'.format(caption, name_1, name_2, depth_vit, d, hid)
            # cross_mat_img = 'flows/heatmaps/cross_flow_face_mat_{}_{}_vs_{}_depth_{}_did_{}_head_{}.jpg'.format(caption, name_1, name_2, depth_vit, d, hid)
        
            # print('cross img: {}'.format(cross_imgname))
            # print('cross mat flow img: {}'.format(cross_mat_img))

            image.save(heatmap_imgname)
            # cross_mat.save(cross_mat_img)

    print('done')
