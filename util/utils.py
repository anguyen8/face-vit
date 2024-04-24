import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

from .verification import evaluate, evaluate_transmatcher, evaluate_emd

from datetime import datetime
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
from PIL import Image
import mxnet as mx
import io
import os, pickle, sklearn
import time
from IPython import embed
import cv2
from .emd import emd_similarity 

def get_time():
    return (str(datetime.now())[:-10]).replace(' ', '-').replace(':', '-')


def load_bin(path, image_size=[112,112]):
    bins, issame_list = pickle.load(open(path, 'rb'), encoding='bytes')
    data_list = []
    for flip in [0,1]:
        data = torch.zeros((len(issame_list)*2, 3, image_size[0], image_size[1]))
        data_list.append(data)
    for i in range(len(issame_list)*2):
        _bin = bins[i]
        img = mx.image.imdecode(_bin)
        if img.shape[1]!=image_size[0]:
            img = mx.image.resize_short(img, image_size[0])
        img = mx.nd.transpose(img, axes=(2, 0, 1))
        for flip in [0,1]:
            if flip==1:
                img = mx.ndarray.flip(data=img, axis=2)
            data_list[flip][i][:] = torch.tensor(img.asnumpy())
        if i%1000==0:
            print('loading bin', i)
    print(data_list[0].shape)
    return data_list, issame_list


def get_val_pair(path, name):
    ver_path = os.path.join(path,name + ".bin")
    print(ver_path)
    assert os.path.exists(ver_path)
    data_set, issame = load_bin(ver_path)
    print('ver', name)
    return data_set, issame


def get_val_data(data_path, targets):
    assert len(targets) > 0
    vers = []
    for t in targets:
        data_set, issame = get_val_pair(data_path, t)
        vers.append([t, data_set, issame])
    return vers


def separate_irse_bn_paras(modules):
    if not isinstance(modules, list):
        modules = [*modules.modules()]
    paras_only_bn = []
    paras_wo_bn = []
    for layer in modules:
        if 'model' in str(layer.__class__):
            continue
        if 'container' in str(layer.__class__):
            continue
        else:
            if 'batchnorm' in str(layer.__class__):
                paras_only_bn.extend([*layer.parameters()])
            else:
                paras_wo_bn.extend([*layer.parameters()])

    return paras_only_bn, paras_wo_bn


def separate_resnet_bn_paras(modules):
    all_parameters = modules.parameters()
    paras_only_bn = []

    for pname, p in modules.named_parameters():
        if pname.find('bn') >= 0:
            paras_only_bn.append(p)
            
    paras_only_bn_id = list(map(id, paras_only_bn))
    paras_wo_bn = list(filter(lambda p: id(p) not in paras_only_bn_id, all_parameters))
    
    return paras_only_bn, paras_wo_bn


def separate_mobilefacenet_bn_paras(modules):
    if not isinstance(modules, list):
        modules = [*modules.modules()]
    paras_only_bn = []
    paras_wo_bn = []
    for layer in modules:
        if 'mobilefacenet' in str(layer.__class__) or 'container' in str(layer.__class__):
            continue
        if 'batchnorm' in str(layer.__class__):
            paras_only_bn.extend([*layer.parameters()])
        else:
            paras_wo_bn.extend([*layer.parameters()])

    return paras_only_bn, paras_wo_bn


def gen_plot(fpr, tpr):
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    plt.xlabel("FPR", fontsize = 14)
    plt.ylabel("TPR", fontsize = 14)
    plt.title("ROC Curve", fontsize = 14)
    plot = plt.plot(fpr, tpr, linewidth = 2)
    buf = io.BytesIO()
    plt.savefig(buf, format = 'jpeg')
    buf.seek(0)
    plt.close()

    return buf

def test_forward(device, backbone, data_set):
    backbone = backbone.to(device)
    backbone.eval() # switch to evaluation mode
    #embed()
    #last_time1 = time.time()
    forward_time = 0
    carray = data_set[0]
        #print("carray:",carray.shape)
    idx = 0
    with torch.no_grad():
            while idx < 2000:
                batch = carray[idx:idx + 1]
                batch_device = batch.to(device)
                last_time = time.time()
                backbone(batch_device)
                forward_time += time.time() - last_time
                #if idx % 1000 ==0:
                #    print(idx, forward_time)
                idx += 1
    print("forward_time", 2000, forward_time, 2000/forward_time)
    return forward_time

def perform_val_merge_test_resnet(multi_gpu, device, embedding_size, batch_size, backbone, data_set, issame, nrof_folds = 10):
    if multi_gpu:
        backbone = backbone.module # unpackage model from DataParallel
        backbone = backbone.to(device)
    else:
        backbone = backbone.to(device)
    backbone.eval() # switch to evaluation mode

    embeddings_list = []
    
    array1, array2 = data_set
    idx = 0
    N = len(array1)
    h = int(N/2)
    embeddings1 = np.zeros([N, embedding_size])
    embeddings2 = np.zeros([N, embedding_size])
    batch_11 = array1[0::2] 
    batch_12 = array1[1::2]
    batch_21 = array2[0::2]
    batch_22 = array2[1::2]

    with torch.no_grad():
        while idx + batch_size <= h: #len(array1):
            batch1 = batch_11[idx:idx + batch_size] 
            batch2 = batch_12[idx:idx + batch_size] 
            inputs = torch.cat([batch1, batch2], 0)
            outputs = backbone(inputs.to(device)) #backbone(inputs.to(device), fea=True)
            embeddings1[0::2][idx:idx + batch_size] = outputs['fea'][0:batch_size].cpu() 
            embeddings1[1::2][idx:idx + batch_size] = outputs['fea'][batch_size:].cpu() 
            
            batch1 = batch_21[idx:idx + batch_size] 
            batch2 = batch_22[idx:idx + batch_size] 
            inputs = torch.cat([batch1, batch2], 0)
            outputs = backbone(inputs.to(device))
            embeddings2[0::2][idx:idx + batch_size] = outputs['fea'][0:batch_size].cpu() 
            embeddings2[1::2][idx:idx + batch_size] = outputs['fea'][batch_size:].cpu() 
            # embeddings2[idx:idx + batch_size] = outputs[1].cpu()
            idx += batch_size
        if idx < h: #len(array1):
            batch1 = batch_11[idx:]
            batch2 = batch_12[idx:]
            R, _, _, _ = batch1.size()
            inputs = torch.cat([batch1, batch2], 0)
            outputs = backbone(inputs.to(device))
            embeddings1[0::2][idx:] = outputs['fea'][0:R].cpu() 
            embeddings1[1::2][idx:] = outputs['fea'][R:].cpu() 

            batch1 = batch_21[idx:]
            batch2 = batch_22[idx:]
            inputs = torch.cat([batch1, batch2], 0)
            outputs = backbone(inputs.to(device))
            embeddings2[0::2][idx:] = outputs['fea'][0:R].cpu() 
            embeddings2[1::2][idx:] = outputs['fea'][R:].cpu() 

    embeddings_list.append(embeddings1)
    embeddings_list.append(embeddings2)

    _xnorm = 0.0
    _xnorm_cnt = 0
    for embed in embeddings_list:
        for i in range(embed.shape[0]):
            _em = embed[i]
            _norm=np.linalg.norm(_em)
            _xnorm+=_norm
            _xnorm_cnt+=1
    _xnorm /= _xnorm_cnt

    embeddings = embeddings_list[0] + embeddings_list[1]
    embeddings = sklearn.preprocessing.normalize(embeddings)
    print(embeddings.shape)

    tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds)
    buf = gen_plot(fpr, tpr)
    roc_curve = Image.open(buf)
    roc_curve_tensor = transforms.ToTensor()(roc_curve)

    return accuracy.mean(), accuracy.std(), _xnorm, best_thresholds.mean(), roc_curve_tensor

def process_img(img_path, grayscale=False, size=112, use_scale=True):
    if grayscale:
        img = cv2.imread(img_path, 0)
        c = 1
    else:
        if use_scale:
            img = cv2.imread(img_path)
        else:
            img = cv2.imread(img_path)[:,:,::-1]
        c = 3

    img = cv2.resize(img, (size,size))
    img = img.reshape((size,size,c))
    img = img.transpose((2, 0, 1))
    img = img.astype(np.float32, copy=False)
    if use_scale:
        img -= 127.5
        img /= 127.5
    img = torch.from_numpy(img).float()
    return img

def perform_val_color_images_transmatcher(multi_gpu, device, embedding_size, batch_size, backbone, matcher, grayscale=False, size=112, nrof_folds = 10):
    if multi_gpu:
        backbone = backbone.module # unpackage model from DataParallel
        backbone = backbone.to(device)
    else:
        backbone = backbone.to(device)
    backbone.eval() # switch to evaluation mode
    matcher.eval()
    embeddings_list = []
    issame = []
    N = 6000
    pair_list = 'lfw_test_pair.txt'
    # data_root = '/home/hai/datasets/lfw_112'
    data_root = '/home/hai/workspace/InsightFace-v2/data/lfw_crop_112'
    # batch1 = np.zeros([batch_size, embedding_size])
    # batch2 = np.zeros([batch_size, embedding_size])
    dist = np.zeros([N, 1])
    accuracy = []
    with open(pair_list, 'r') as fd:
        for i, line in enumerate(fd):
            line = line.strip()
            splits = line.split()
            img_path = os.path.join(data_root, splits[0])
            img1 = process_img(img_path, size=size, grayscale=grayscale)
            img_path = os.path.join(data_root, splits[1])
            img2 = process_img(img_path, size=size, grayscale=grayscale)
            inputs = torch.cat((img1.unsqueeze(0), img2.unsqueeze(0)), 0)
            features = backbone(inputs.to(device))
            matcher.make_kernel(features[0].unsqueeze(0))
            score = matcher(features[1].unsqueeze(0))
            score = torch.sigmoid(score / 10)
            score = score.cpu()  #(1.0-score).cpu()
            dist[i] = score.item()
            issame.append(float(splits[-1]))

    _xnorm = 0.0
    tpr, fpr, accuracy, best_thresholds = evaluate_transmatcher(dist, issame, nrof_folds)
    buf = gen_plot(fpr, tpr)
    roc_curve = Image.open(buf)
    roc_curve_tensor = transforms.ToTensor()(roc_curve)

    return accuracy.mean(), accuracy.std(), _xnorm, best_thresholds.mean(), roc_curve_tensor

def perform_val_emd_color_images(multi_gpu, device, embedding_size, batch_size, backbone, grayscale=False, size=112, nrof_folds = 10, target='lfw'):
    if multi_gpu:
        backbone = backbone.module # unpackage model from DataParallel
        backbone = backbone.to(device)
    else:
        backbone = backbone.to(device)
    backbone.eval() # switch to evaluation mode

    embeddings_list = []
    issame = []
    N = 6000
    embeddings = np.zeros([2*N, embedding_size])
    feature_no_avg_bank = []
    avgpool_bank_center = []

    if target == 'lfw':
        pair_list = 'lfw_test_pair.txt'
        data_root = '/home/hai/workspace/InsightFace-v2/data/lfw_crop_112'
    elif target == 'talfw':
        pair_list = 'lfw_test_pair.txt'
        data_root = '/home/hai/datasets/cropped_TALFW_128x128'
    else:
        pair_list = '/home/hai/datasets/MLFW/pairs.txt'
        data_root = '/home/hai/datasets/MLFW/aligned'

    with open(pair_list, 'r') as fd:
        for i, line in enumerate(fd):
            line = line.strip()
            splits = line.split()
            img_path = os.path.join(data_root, splits[0])
            img1 = process_img(img_path, size=size, grayscale=grayscale)
            img_path = os.path.join(data_root, splits[1])
            img2 = process_img(img_path, size=size, grayscale=grayscale)
            inputs = torch.cat((img1.unsqueeze(0), img2.unsqueeze(0)), 0)
            outputs = backbone(inputs.to(device))
            no_avg_feat = outputs['embedding_88'] 
            avg_pool = outputs['adpt_pooling_88']
            avgpool_bank_center.append(avg_pool.data)
            feature_no_avg_bank.append(no_avg_feat.data)
            embeddings[2*i] = outputs['fea'][0].squeeze(0).cpu().detach().numpy() #outputs[0].squeeze(0).cpu().detach().numpy()
            embeddings[2*i + 1] = outputs['fea'][1].squeeze(0).cpu().detach().numpy() #outputs[1].squeeze(0).cpu().detach().numpy()
            issame.append(float(splits[-1]))
    
    feature_no_avg_bank = torch.cat(feature_no_avg_bank, dim=0)
    N, C, _, _ = feature_no_avg_bank.size()
    feature_no_avg_bank = feature_no_avg_bank.view(N, C, -1)
    avgpool_bank_center = torch.cat(avgpool_bank_center, dim=0).squeeze(-1).squeeze(-1)
    
    _xnorm = 0.0
    embeddings = sklearn.preprocessing.normalize(embeddings)
    print(embeddings.shape)

    feature_no_avg_bank = torch.nn.functional.normalize(feature_no_avg_bank, p=2, dim=1)
    avgpool_bank_center = torch.nn.functional.normalize(avgpool_bank_center, p=2, dim=1)
    
    emd_inputs = {'feature_no_avg_bank': feature_no_avg_bank,
                 'avgpool_bank_center':avgpool_bank_center}

    tpr, fpr, accuracy, best_thresholds = evaluate_emd(embeddings, emd_inputs, issame, nrof_folds, alpha=0.7)
    buf = gen_plot(fpr, tpr)
    roc_curve = Image.open(buf)
    roc_curve_tensor = transforms.ToTensor()(roc_curve)

    return accuracy.mean(), accuracy.std(), _xnorm, best_thresholds.mean(), roc_curve_tensor

# For grayscal image of arcface resnet 18 only
def perform_val_color_images_hybrid_vit(multi_gpu, device, embedding_size, batch_size, backbone, grayscale=False, size=112, nrof_folds = 10, use_scale=True, target='lfw'):
    if multi_gpu:
        backbone = backbone.module # unpackage model from DataParallel
        backbone = backbone.to(device)
    else:
        backbone = backbone.to(device)
    backbone.eval() # switch to evaluation mode

    embeddings_list = []
    issame = []
    N = 6000
    embeddings = np.zeros([2*N, embedding_size])
    if target == 'lfw':
        pair_list = 'lfw_test_pair.txt'
        data_root = '/home/hai/workspace/InsightFace-v2/data/lfw_crop_112'
    elif target == 'glfw':
        pair_list = 'glfw_test_pair.txt'
        data_root = '/home/hai/workspace/lfw-align-128-glass'
    elif target == 'talfw':
        pair_list = 'lfw_test_pair.txt'
        data_root = '/home/hai/datasets/cropped_TALFW_128x128'
    else:
        pair_list = '/home/hai/datasets/MLFW/pairs.txt'
        data_root = '/home/hai/datasets/MLFW/aligned'

    with open(pair_list, 'r') as fd:
        for i, line in enumerate(fd):
            line = line.strip()
            splits = line.split()
            img_path = os.path.join(data_root, splits[0])
            # print(img_path)
            img1 = process_img(img_path, size=size, grayscale=grayscale, use_scale=use_scale)
            img_path = os.path.join(data_root, splits[1])
            # print(img_path)
            img2 = process_img(img_path, size=size, grayscale=grayscale, use_scale=use_scale)
            inputs = torch.cat((img1.unsqueeze(0), img2.unsqueeze(0)), 0)
            outputs = backbone(inputs.to(device))
            # outputs = backbone(inputs.to(device))
            embeddings[2*i] = outputs[0].squeeze(0).cpu().detach().numpy()
            embeddings[2*i + 1] = outputs[1].squeeze(0).cpu().detach().numpy()
            issame.append(float(splits[-1]))
    
    _xnorm = 0.0
    embeddings = sklearn.preprocessing.normalize(embeddings)
    print(embeddings.shape)

    tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds)
    buf = gen_plot(fpr, tpr)
    roc_curve = Image.open(buf)
    roc_curve_tensor = transforms.ToTensor()(roc_curve)

    return accuracy.mean(), accuracy.std(), _xnorm, best_thresholds.mean(), roc_curve_tensor

# For grayscal image of arcface resnet 18 only
def perform_val_color_images(multi_gpu, device, embedding_size, batch_size, backbone, grayscale=False, size=112, nrof_folds = 10, use_scale=True, vit=False, target='lfw'):
    if multi_gpu:
        backbone = backbone.module # unpackage model from DataParallel
        backbone = backbone.to(device)
    else:
        backbone = backbone.to(device)
    backbone.eval() # switch to evaluation mode

    embeddings_list = []
    issame = []
    N = 6000
    embeddings = np.zeros([2*N, embedding_size])
    if target == 'lfw':
        pair_list = 'lfw_test_pair.txt'
        data_root = '/home/hai/workspace/InsightFace-v2/data/lfw_crop_112'
    elif target == 'talfw':
        pair_list = 'lfw_test_pair.txt'
        data_root = '/home/hai/datasets/cropped_TALFW_128x128'
    else:
        pair_list = '/home/hai/datasets/MLFW/pairs.txt'
        data_root = '/home/hai/datasets/MLFW/aligned'

    with open(pair_list, 'r') as fd:
        for i, line in enumerate(fd):
            line = line.strip()
            splits = line.split()
            img_path = os.path.join(data_root, splits[0])
            img1 = process_img(img_path, size=size, grayscale=grayscale, use_scale=use_scale)
            img_path = os.path.join(data_root, splits[1])
            img2 = process_img(img_path, size=size, grayscale=grayscale, use_scale=use_scale)
            inputs = torch.cat((img1.unsqueeze(0), img2.unsqueeze(0)), 0)
            if vit:
                outputs = backbone(inputs.to(device))
            else:
                outputs = backbone(inputs.to(device), fea=True)
            
            embeddings[2*i] = outputs[0].squeeze(0).cpu().detach().numpy()
            embeddings[2*i + 1] = outputs[1].squeeze(0).cpu().detach().numpy()
            issame.append(float(splits[-1]))
    
    _xnorm = 0.0
    embeddings = sklearn.preprocessing.normalize(embeddings)
    print(embeddings.shape)

    tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds)
    buf = gen_plot(fpr, tpr)
    roc_curve = Image.open(buf)
    roc_curve_tensor = transforms.ToTensor()(roc_curve)

    return accuracy.mean(), accuracy.std(), _xnorm, best_thresholds.mean(), roc_curve_tensor

def perform_val_color_images_cls(multi_gpu, device, embedding_size, batch_size, backbone, grayscale=False, size=112, nrof_folds = 10, use_scale=True, target='lfw'):
    if multi_gpu:
        backbone = backbone.module # unpackage model from DataParallel
        backbone = backbone.to(device)
    else:
        backbone = backbone.to(device)
    backbone.eval() # switch to evaluation mode

    embeddings_list = []
    issame = []
    outs = []
    N = 6000
    embeddings = np.zeros([2*N, embedding_size])
    if target == 'lfw':
        pair_list = 'lfw_test_pair.txt'
        data_root = '/home/hai/workspace/InsightFace-v2/data/lfw_crop_112'
    elif target == 'talfw':
        pair_list = 'lfw_test_pair.txt'
        data_root = '/home/hai/datasets/cropped_TALFW_128x128'
    else:
        pair_list = '/home/hai/datasets/MLFW/pairs.txt'
        data_root = '/home/hai/datasets/MLFW/aligned'

    
    with open(pair_list, 'r') as fd:
        for i, line in enumerate(fd):
            line = line.strip()
            splits = line.split()
            img_path = os.path.join(data_root, splits[0])
            img1 = process_img(img_path, size=size, grayscale=grayscale, use_scale=use_scale)
            img_path = os.path.join(data_root, splits[1])
            img2 = process_img(img_path, size=size, grayscale=grayscale, use_scale=use_scale)
            inputs = torch.cat((img1.unsqueeze(0), img2.unsqueeze(0)), 0)
            outputs = backbone(inputs.to(device), fea=True)
            outs.append(outputs)
            # if outputs
            issame.append(float(splits[-1]))
    
    accuracy = 0.0
    for out, re in zip(outs, issame):
        if out == re:
            accuracy += 1.0
    
    return accuracy / N
    
    # _xnorm = 0.0
    # embeddings = sklearn.preprocessing.normalize(embeddings)
    # print(embeddings.shape)

    # tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds)
    # buf = gen_plot(fpr, tpr)
    # roc_curve = Image.open(buf)
    # roc_curve_tensor = transforms.ToTensor()(roc_curve)

    # return accuracy.mean(), accuracy.std(), _xnorm, best_thresholds.mean(), roc_curve_tensor

def perform_val_resnet_color_images(multi_gpu, device, embedding_size, batch_size, backbone, grayscale=False, size=112, nrof_folds = 10, use_scale=True, target='lfw'):
    if multi_gpu:
        backbone = backbone.module # unpackage model from DataParallel
        backbone = backbone.to(device)
    else:
        backbone = backbone.to(device)
    backbone.eval() # switch to evaluation mode

    embeddings_list = []
    issame = []
    N = 6000
    embeddings = np.zeros([2*N, embedding_size])
    # pair_list = 'lfw_test_pair.txt'
    # data_root = '/home/hai/workspace/InsightFace-v2/data/lfw_crop_112'

    if target == 'lfw':
        pair_list = 'lfw_test_pair.txt'
        data_root = '/home/hai/workspace/InsightFace-v2/data/lfw_crop_112'
    elif target == 'talfw':
        pair_list = 'lfw_test_pair.txt'
        data_root = '/home/hai/datasets/cropped_TALFW_128x128'
    else:
        pair_list = '/home/hai/datasets/MLFW/pairs.txt'
        data_root = '/home/hai/datasets/MLFW/aligned'

    # pair_list = '/home/hai/datasets/MLFW/pairs.txt'
    # data_root = '/home/hai/datasets/MLFW/aligned'

    with open(pair_list, 'r') as fd:
        for i, line in enumerate(fd):
            line = line.strip()
            splits = line.split()
            img_path = os.path.join(data_root, splits[0])
            img1 = process_img(img_path, size=size, grayscale=grayscale)
            img_path = os.path.join(data_root, splits[1])
            img2 = process_img(img_path, size=size, grayscale=grayscale)
            inputs = torch.cat((img1.unsqueeze(0), img2.unsqueeze(0)), 0)
            outputs = backbone(inputs.to(device))['fea']
            embeddings[2*i] = outputs[0].squeeze(0).cpu().detach().numpy()
            embeddings[2*i + 1] = outputs[1].squeeze(0).cpu().detach().numpy()
            issame.append(float(splits[-1]))
    
    _xnorm = 0.0
    embeddings = sklearn.preprocessing.normalize(embeddings)
    print(embeddings.shape)

    tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds)
    buf = gen_plot(fpr, tpr)
    roc_curve = Image.open(buf)
    roc_curve_tensor = transforms.ToTensor()(roc_curve)

    return accuracy.mean(), accuracy.std(), _xnorm, best_thresholds.mean(), roc_curve_tensor

def perform_val_merge(multi_gpu, device, embedding_size, batch_size, backbone, data_set, issame, nrof_folds = 10):
    if multi_gpu:
        backbone = backbone.module # unpackage model from DataParallel
        backbone = backbone.to(device)
    else:
        backbone = backbone.to(device)
    backbone.eval() # switch to evaluation mode

    embeddings_list = []
    
    array1, array2 = data_set
    idx = 0
    N = len(array1)
    h = int(N/2)
    embeddings1 = np.zeros([N, embedding_size])
    embeddings2 = np.zeros([N, embedding_size])
    batch_11 = array1[0::2] 
    batch_12 = array1[1::2]
    batch_21 = array2[0::2]
    batch_22 = array2[1::2]

    with torch.no_grad():
        while idx + batch_size <= h: #len(array1):
            batch1 = batch_11[idx:idx + batch_size] 
            batch2 = batch_12[idx:idx + batch_size] 
            inputs = torch.cat([batch1, batch2], 0)
            outputs = backbone(inputs.to(device), fea=True)
            embeddings1[0::2][idx:idx + batch_size] = outputs[0].cpu() 
            embeddings1[1::2][idx:idx + batch_size] = outputs[1].cpu() 
            
            batch1 = batch_21[idx:idx + batch_size] 
            batch2 = batch_22[idx:idx + batch_size] 
            inputs = torch.cat([batch1, batch2], 0)
            outputs = backbone(inputs.to(device), fea=True)
            embeddings2[0::2][idx:idx + batch_size] = outputs[0].cpu() 
            embeddings2[1::2][idx:idx + batch_size] = outputs[1].cpu() 
            # embeddings2[idx:idx + batch_size] = outputs[1].cpu()
            idx += batch_size
        if idx < h: #len(array1):
            batch1 = batch_11[idx:]
            batch2 = batch_12[idx:]
            inputs = torch.cat([batch1, batch2], 0)
            outputs = backbone(inputs.to(device), fea=True)
            embeddings1[0::2][idx:] = outputs[0].cpu() 
            embeddings1[1::2][idx:] = outputs[1].cpu() 

            batch1 = batch_21[idx:]
            batch2 = batch_22[idx:]
            inputs = torch.cat([batch1, batch2], 0)
            outputs = backbone(inputs.to(device), fea=True)
            embeddings2[0::2][idx:] = outputs[0].cpu() 
            embeddings2[1::2][idx:] = outputs[1].cpu() 

    embeddings_list.append(embeddings1)
    embeddings_list.append(embeddings2)

    _xnorm = 0.0
    _xnorm_cnt = 0
    for embed in embeddings_list:
        for i in range(embed.shape[0]):
            _em = embed[i]
            _norm=np.linalg.norm(_em)
            _xnorm+=_norm
            _xnorm_cnt+=1
    _xnorm /= _xnorm_cnt

    embeddings = embeddings_list[0] + embeddings_list[1]
    embeddings = sklearn.preprocessing.normalize(embeddings)
    print(embeddings.shape)

    tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds)
    buf = gen_plot(fpr, tpr)
    roc_curve = Image.open(buf)
    roc_curve_tensor = transforms.ToTensor()(roc_curve)

    return accuracy.mean(), accuracy.std(), _xnorm, best_thresholds.mean(), roc_curve_tensor

def perform_val(multi_gpu, device, embedding_size, batch_size, backbone, data_set, issame, nrof_folds = 10):
    if multi_gpu:
        backbone = backbone.module # unpackage model from DataParallel
        backbone = backbone.to(device)
    else:
        backbone = backbone.to(device)
    backbone.eval() # switch to evaluation mode

    embeddings_list = []
    for carray in data_set:
        idx = 0
        embeddings = np.zeros([len(carray), embedding_size])
        with torch.no_grad():
            while idx + batch_size <= len(carray):
                batch = carray[idx:idx + batch_size]
                #last_time = time.time()
                # out = backbone(batch.to(device))
                # embeddings[idx:idx + batch_size] = out['fea'].cpu()
                embeddings[idx:idx + batch_size] = backbone(batch.to(device)).cpu()
                #batch_time = time.time() - last_time
                #print("batch_time", batch_size, batch_time)
                idx += batch_size
            if idx < len(carray):
                batch = carray[idx:]
                # out = backbone(batch.to(device))
                # embeddings[idx:] = out['fea'].cpu()
                out = backbone(batch.to(device)).cpu()
        embeddings_list.append(embeddings)

    _xnorm = 0.0
    _xnorm_cnt = 0
    for embed in embeddings_list:
        for i in range(embed.shape[0]):
            _em = embed[i]
            _norm=np.linalg.norm(_em)
            _xnorm+=_norm
            _xnorm_cnt+=1
    _xnorm /= _xnorm_cnt

    embeddings = embeddings_list[0] + embeddings_list[1]
    embeddings = sklearn.preprocessing.normalize(embeddings)
    print(embeddings.shape)

    tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds)
    buf = gen_plot(fpr, tpr)
    roc_curve = Image.open(buf)
    roc_curve_tensor = transforms.ToTensor()(roc_curve)

    return accuracy.mean(), accuracy.std(), _xnorm, best_thresholds.mean(), roc_curve_tensor

def perform_val_deit(multi_gpu, device, embedding_size, batch_size, backbone, dis_token, data_set, issame, nrof_folds = 10):
    if multi_gpu:
        backbone = backbone.module # unpackage model from DataParallel
        backbone = backbone.to(device)
    else:
        backbone = backbone.to(device)
    backbone.eval() # switch to evaluation mode

    embeddings_list = []
    for carray in data_set:
        idx = 0
        embeddings = np.zeros([len(carray), embedding_size])
        with torch.no_grad():
            while idx + batch_size <= len(carray):
                batch = carray[idx:idx + batch_size]
                #last_time = time.time()
                #embed()
                fea,token = backbone(batch.to(device), dis_token.to(device))
                embeddings[idx:idx + batch_size] = fea.cpu()
                #batch_time = time.time() - last_time
                #print("batch_time", batch_size, batch_time)
                idx += batch_size
            if idx < len(carray):
                batch = carray[idx:]
                embeddings[idx:] = backbone(batch.to(device)).cpu()
        embeddings_list.append(embeddings)

    _xnorm = 0.0
    _xnorm_cnt = 0
    for embed in embeddings_list:
        for i in range(embed.shape[0]):
            _em = embed[i]
            _norm=np.linalg.norm(_em)
            _xnorm+=_norm
            _xnorm_cnt+=1
    _xnorm /= _xnorm_cnt

    embeddings = embeddings_list[0] + embeddings_list[1]
    embeddings = sklearn.preprocessing.normalize(embeddings)
    print(embeddings.shape)

    tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds)
    buf = gen_plot(fpr, tpr)
    roc_curve = Image.open(buf)
    roc_curve_tensor = transforms.ToTensor()(roc_curve)

    return accuracy.mean(), accuracy.std(), _xnorm, best_thresholds.mean(), roc_curve_tensor

def buffer_val(writer, db_name, acc, std, xnorm, best_threshold, roc_curve_tensor, batch):
    writer.add_scalar('Accuracy/{}_Accuracy'.format(db_name), acc, batch)
    writer.add_scalar('Std/{}_Std'.format(db_name), std, batch)
    writer.add_scalar('XNorm/{}_XNorm'.format(db_name), xnorm, batch)
    writer.add_scalar('Threshold/{}_Best_Threshold'.format(db_name), best_threshold, batch)
    writer.add_image('ROC/{}_ROC_Curve'.format(db_name), roc_curve_tensor, batch)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val   = 0
        self.avg   = 0
        self.sum   = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val   = val
        self.sum   += val * n
        self.count += n
        self.avg   = self.sum / self.count

'''
def train_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred    = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res
'''

def train_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred    = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    #embed()
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res[0]
