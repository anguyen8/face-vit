import wandb
import os, argparse, sklearn
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tensorboardX import SummaryWriter

from config import get_config
from dataset import FaceDataset, FacePairDataset, FacePairLabelDataset, FaceImgDataset

# from util.utils import separate_irse_bn_paras, separate_resnet_bn_paras, separate_mobilefacenet_bn_paras
from util.utils import get_val_data, perform_val_merge, perform_val_color_images_cls, perform_val_color_images, perform_val_color_images_hybrid_vit, get_time, buffer_val, AverageMeter, train_accuracy

import time
from models import ViT_face
from models import ViTs_face
from models.vit_model_face import ViT_face_model, Hybrid_ViT
from models.model_irse import IR_50
from models.resnet import resnet_face18
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer

def need_save(acc, highest_acc):
    do_save = False
    save_cnt = 0
    if acc[0] > 0.98:
        do_save = True
    for i, accuracy in enumerate(acc):
        if accuracy > highest_acc[i]:
            highest_acc[i] = accuracy
            do_save = True
        if i > 0 and accuracy >= highest_acc[i]-0.002:
            save_cnt += 1
    if save_cnt >= len(acc)*3/4 and acc[0]>0.99:
        do_save = True
    print("highest_acc:", highest_acc)
    return do_save

def arrange_label(labels):
    N = labels.size()[0]
    half = int(N/2)
    splits = torch.split(labels, half)
    new_labels = splits[0] == splits[1]
    return new_labels.float()

depth_vit = 8
num_heads = 1
grayscale = True
use_scale = True
size = 128
facemodel = 'ArcFace'
database = 'casia'
# wandb.init(project="face-transformer-facemodel-{}-depth-{}-heads-{}-resnet18-gray_lr_1e5_remove_sep".format(facemodel, depth_vit, num_heads), entity="name")
# wandb.init(project="face-transformer-facemodel-{}-depth-{}-heads-{}-resnet18-gray_fc1024_lr_1e5_dropout_0_LFW".format(facemodel, depth_vit, num_heads), entity="name")
wandb.init(project="ViT_P8S8_2-image-facemodel-{}-{}-depth-{}-heads-{}-fc512_lr_1e5_hybrid_vit".format(database, facemodel, depth_vit, num_heads), entity="name")
# wandb.init(project="ViT_P8S8_2-image-facemodel-{}-{}-depth-{}-heads-{}-resnet18-gray_fc512_lr_LFW_1e5".format(database, facemodel, depth_vit, num_heads), entity="name")
# wandb.init(project="face-transformer-facemodel-depth-{}-resnet50".format(depth_vit), entity="name")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-w", "--workers_id", help="gpu ids or cpu", default='cpu', type=str)
    parser.add_argument("-e", "--epochs", help="training epochs", default=125, type=int)
    parser.add_argument("-b", "--batch_size", help="batch_size", default=256, type=int)
    parser.add_argument("-d", "--data_mode", help="use which database, [casia, vgg, ms1m, retina, ms1mr]",default='ms1m', type=str)
    parser.add_argument("-n", "--net", help="which network, ['VIT','VITs']",default='VITs', type=str)
    parser.add_argument("-head", "--head", help="head type, ['Softmax', 'ArcFace', 'CosFace', 'SFaceLoss']", default='ArcFace', type=str)
    parser.add_argument("-t", "--target", help="verification targets", default='lfw,talfw,calfw,cplfw,cfp_fp,agedb_30', type=str)
    parser.add_argument("-r", "--resume", help="resume model", default='', type=str)
    parser.add_argument('--outdir', help="output dir", default='', type=str)
    parser.add_argument('-use_cls', action='store_true', help="Use classification",)
    parser.add_argument('-no_ver', action='store_true', help="Do not do verification",)
    parser.add_argument('-use_face_loss', action='store_true', help="Do not do verification",)
    parser.add_argument('-no_face_model', action='store_true', help="Do not use face model",)
    parser.add_argument('-single_mlp', action='store_true', help="use single mlp",)
    parser.add_argument('-remove_sep', action='store_true', help="remove seperate",)
    parser.add_argument('-remove_pos', action='store_true', help="remove position embeddings",)
    parser.add_argument('-use_wandb', action='store_true', help="Use wandb server",)
    parser.add_argument('-out_dim', type=int, default=512, metavar='N',
                        help='output feature dim')

    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
                        help='learning rate (default: 1e-5)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')
    args = parser.parse_args()

    #======= hyperparameters & data loaders =======#
    cfg = get_config(args)

    SEED = cfg['SEED'] # random seed for reproduce results
    torch.manual_seed(SEED)

    DATA_ROOT = cfg['DATA_ROOT'] # the parent root where your train/val/test data are stored
    EVAL_PATH = cfg['EVAL_PATH']
    WORK_PATH = cfg['WORK_PATH'] # the root to buffer your checkpoints and to log your train/val status
    BACKBONE_RESUME_ROOT = cfg['BACKBONE_RESUME_ROOT'] # the root to resume training from a saved checkpoint

    BACKBONE_NAME = cfg['BACKBONE_NAME']
    HEAD_NAME = cfg['HEAD_NAME'] # support:  ['Softmax', 'ArcFace', 'CosFace', 'SFaceLoss']

    INPUT_SIZE = cfg['INPUT_SIZE']
    EMBEDDING_SIZE = args.out_dim #cfg['EMBEDDING_SIZE'] # feature dimension
    BATCH_SIZE = cfg['BATCH_SIZE']
    NUM_EPOCH = cfg['NUM_EPOCH']

    DEVICE = cfg['DEVICE']
    MULTI_GPU = cfg['MULTI_GPU'] # flag to use multiple GPUs
    GPU_ID = cfg['GPU_ID'] # specify your GPU ids
    print('args: ', args)
    print('GPU_ID', GPU_ID)
    TARGET = ['lfw'] #cfg['TARGET']
    print("=" * 60)
    print("Overall Configurations:")
    print(cfg)
    with open(os.path.join(WORK_PATH, 'config.txt'), 'w') as f:
        f.write(str(cfg))
    print("=" * 60)

    writer = SummaryWriter(WORK_PATH) # writer for buffering intermedium results
    torch.backends.cudnn.benchmark = True
    best_acc = -1.0
    # with open(os.path.join(DATA_ROOT, 'property'), 'r') as f:
    #     NUM_CLASS, h, w = [int(i) for i in f.read().split(',')]
    # assert h == INPUT_SIZE[0] and w == INPUT_SIZE[1]
    # if args.use_face_loss:
    if args.use_cls:
        NUM_CLASS = 10575
        data_dir = '/home/hai/datasets/casia-112x112'
        label_file = 'casia_pair_112_2m.txt'
    else:
        if database == 'casia':
            NUM_CLASS = 10575
            data_dir = '/home/hai/datasets/casia-112x112'
            label_file = 'casia_pair_112_label_2m.txt'
        else:
            NUM_CLASS = 93431
            data_dir = '/home/hai/datasets/ms1m-retinaface-t1/images' #'/home/hai/datasets/casia-112x112'
            label_file = 'ms_celeb_pair_112_label_2m.txt' 

    channels = 3
    # dataset = FaceDataset(os.path.join(DATA_ROOT, 'train.rec'), rand_mirror=True)
    if args.use_face_loss:
        if args.no_face_model == False:
            dataset = FacePairLabelDataset(data_dir, label_file, size=size, grayscale=grayscale, use_scale=use_scale)
            channels = 1
        else:
            dataset = FacePairLabelDataset(data_dir, label_file, size=size, grayscale=grayscale, use_scale=use_scale)
    else:
        if BACKBONE_NAME == 'Hybrid_ViT':
            dataset = FaceImgDataset('casia_112x112.txt', size=size, grayscale=grayscale, use_scale=use_scale)
        else:
            dataset = FacePairDataset('/home/hai/datasets/casia-112x112', 'casia_pair_112_2m.txt', size=size, grayscale=grayscale, use_scale=use_scale)

    trainloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=len(GPU_ID), drop_last=True)

    print("Number of Training Classes: {}".format(NUM_CLASS))

    # if args.no_ver:
    #     vers = []
    # else:
    vers = ['lfw']
    # vers = get_val_data(EVAL_PATH, TARGET)
    # vers = ['{}'.format(TARGET)] #get_val_data(EVAL_PATH, TARGET)
    highest_acc = [0.0 for t in TARGET]

    wandb.config = {
        "learning_rate": args.lr,
        "epochs": NUM_EPOCH,
        "batch_size": BATCH_SIZE
    }
    #embed()
    #======= model & loss & optimizer =======#
    BACKBONE_DICT = {'VIT': ViT_face(
                         loss_type = HEAD_NAME,
                         GPU_ID = GPU_ID,
                         num_class = NUM_CLASS,
                         image_size=112,
                         patch_size=8,
                         dim=512,
                         depth=20,
                         heads=8,
                         mlp_dim=2048,
                         dropout=0.1,
                         emb_dropout=0.1
                     ),
                     'VITs': ViTs_face(
                         loss_type=HEAD_NAME,
                         GPU_ID=GPU_ID,
                         num_class=NUM_CLASS,
                         image_size=112,
                         patch_size=8,
                         ac_patch_size=12,
                         pad = 4,
                         dim=512,
                         depth=20,
                         heads=8,
                         mlp_dim=2048,
                         dropout=0.1,
                         emb_dropout=0.1
                     ),
                     'VIT_face': ViT_face_model(
                         loss_type=HEAD_NAME,
                         GPU_ID=GPU_ID,
                         num_class=NUM_CLASS,
                         use_cls=args.use_cls,
                         use_face_loss=args.use_face_loss,
                         no_face_model=args.no_face_model,
                         channels=channels,
                         image_size=size,
                         patch_size=8,
                         ac_patch_size=12,
                         pad = 4,
                         dim=512, #256,
                         depth=depth_vit,
                         heads=num_heads,
                         mlp_dim=2048,
                         dropout=0.0,
                         emb_dropout=0.1,
                         out_dim=args.out_dim,
                         singleMLP=args.single_mlp,
                         remove_sep=args.remove_sep,
                         remove_pos=args.remove_pos),
                    'Hybrid_ViT': Hybrid_ViT(
                        loss_type=HEAD_NAME,
                         GPU_ID=GPU_ID,
                         num_class=NUM_CLASS,
                         channels=channels,
                         image_size=size,
                         patch_size=8,
                         ac_patch_size=12,
                         pad = 4,
                         dim=512, #256,
                         depth=depth_vit,
                         heads=num_heads,
                         mlp_dim=2048,
                         dropout=0.1,
                         emb_dropout=0.1,
                         out_dim=args.out_dim, 
                         remove_pos=args.remove_pos),}


    BACKBONE = BACKBONE_DICT[BACKBONE_NAME]
    
    if BACKBONE_NAME in ['VIT_face', 'Hybrid_ViT']  and args.no_face_model == False:
        INPUT_SIZE = [112, 112]
        # facemodel = IR_50(INPUT_SIZE)
        # facemodel.load_state_dict(torch.load('pretrained/Backbone_IR_50.pth'))
        facemodel = resnet_face18(False, use_reduce_pool=False, grayscale=grayscale)
        if grayscale:
            print('Loading ResNet18 Arcface ...')
            model_path =  'pretrained/resnet18_110.pth'
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove module.
                new_state_dict[name] = v
            facemodel.load_state_dict(new_state_dict)

        BACKBONE.face_model = facemodel
        for param in BACKBONE.face_model.parameters():
            param.requires_grad = False

    print("=" * 60)
    print(BACKBONE)
    print("{} Backbone Generated".format(BACKBONE_NAME))
    print("=" * 60)

    # if args.use_cls or args.use_face_loss or :
    LOSS = nn.CrossEntropyLoss()
    # else:
    #     LOSS = nn.BCELoss() #

    #embed()
    OPTIMIZER = create_optimizer(args, BACKBONE)
    print("=" * 60)
    print(OPTIMIZER)
    print("Optimizer Generated")
    print("=" * 60)
    lr_scheduler, _ = create_scheduler(args, OPTIMIZER)

    # optionally resume from a checkpoint
    if BACKBONE_RESUME_ROOT:
        print("=" * 60)
        print(BACKBONE_RESUME_ROOT)
        if os.path.isfile(BACKBONE_RESUME_ROOT):
            print("Loading Backbone Checkpoint '{}'".format(BACKBONE_RESUME_ROOT))
            BACKBONE.load_state_dict(torch.load(BACKBONE_RESUME_ROOT))
        else:
            print("No Checkpoint Found at '{}' . Please Have a Check or Continue to Train from Scratch".format(BACKBONE_RESUME_ROOT))
        print("=" * 60)

    if MULTI_GPU:
        # multi-GPU setting
        BACKBONE = nn.DataParallel(BACKBONE, device_ids = GPU_ID)
        BACKBONE = BACKBONE.to(DEVICE)
    else:
        # single-GPU setting
        BACKBONE = BACKBONE.to(DEVICE)

    #======= train & validation & save checkpoint =======#
    DISP_FREQ = 100 # frequency to display training loss & acc
    VER_FREQ = 500

    # batch = 0  # batch index

    losses = AverageMeter()
    top1 = AverageMeter()


    BACKBONE.train()  # set to training mode
    for epoch in range(NUM_EPOCH): # start training process
        
        lr_scheduler.step(epoch)

        last_time = time.time()
        batch = 0
        # for inputs, labels in iter(trainloader):
        # for inputs1, inputs2, labels in iter(trainloader):
        for samples in iter(trainloader):
            
            if args.use_face_loss:
                inputs1, inputs2, scores, label1, label2 = samples
                labels = torch.cat((label1, label2), dim=0).view(-1)
                labels = labels.to(DEVICE).long()
            else:
                if BACKBONE_NAME == 'Hybrid_ViT':
                    inputs, labels = samples
                else:
                    inputs1, inputs2, labels = samples
                if args.use_cls:
                    labels = labels.to(DEVICE).long()
                else:
                    labels = labels.to(DEVICE).float()
            # compute output
            if BACKBONE_NAME == 'Hybrid_ViT':
                inputs = inputs.to(DEVICE)
            else:
                inputs = torch.cat([inputs1, inputs2], 0).to(DEVICE)
            
                # labels[labels==0] = -1.0

            # outputs, emb = BACKBONE(inputs.float(), labels)
            # labels = arrange_label(labels)
            outputs = BACKBONE(inputs.float(), labels)
            if BACKBONE_NAME == 'Hybrid_ViT':
                labels = labels.squeeze(-1).squeeze(-1).long()
            
            loss = LOSS(outputs, labels)
            #print("outputs", outputs, outputs.data)
            # measure accuracy and record loss
            # prec1= train_accuracy(outputs.data, labels, topk = (1,))

            losses.update(loss.data.item(), inputs.size(0))
            # top1.update(prec1.data.item(), inputs.size(0))


            # compute gradient and do SGD step
            OPTIMIZER.zero_grad()
            loss.backward()
            OPTIMIZER.step()
            
            # dispaly training loss & acc every DISP_FREQ (buffer for visualization)
            if ((batch + 1) % DISP_FREQ == 0) and batch != 0:
                epoch_loss = losses.avg
                epoch_acc = top1.avg
                writer.add_scalar("Training/Training_Loss", epoch_loss, batch + 1)
                writer.add_scalar("Training/Training_Accuracy", epoch_acc, batch + 1)

                batch_time = time.time() - last_time
                last_time = time.time()
                wandb.log({"loss": losses.val})
                print('Epoch {} Batch {}\t'
                      'Speed: {speed:.2f} samples/s\t'
                      'Training Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Training Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    epoch + 1, batch + 1, speed=inputs.size(0) * DISP_FREQ / float(batch_time),
                    loss=losses, top1=top1))
                #print("=" * 60)
                losses = AverageMeter()
                top1 = AverageMeter()

            if ((batch + 1) % VER_FREQ == 0) and batch != 0: #perform validation & save checkpoints (buffer for visualization)
                for params in OPTIMIZER.param_groups:
                    lr = params['lr']
                    break
                print("Learning rate %f"%lr)
                print("Perform Evaluation on", TARGET, ", and Save Checkpoints...")
                acc = []
                for ver in vers:
                    name = ver
                    # accuracy, std, xnorm, best_threshold, roc_curve = perform_val_color_images(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, data_set, issame)
                    if args.no_face_model:
                        name, data_set, issame = ver
                        # accuracy, std, xnorm, best_threshold, roc_curve = perform_val(MULTI_GPU, device, EMBEDDING_SIZE, BATCH_SIZE, model, data_set, issame)
                        accuracy, std, xnorm, best_threshold, roc_curve = perform_val_merge(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, data_set, issame)
                        # accuracy, std, xnorm, best_threshold, roc_curve = perform_val_color_images(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, grayscale=grayscale, size=size)
                    else:
                        if args.use_cls:
                            accuracy = perform_val_color_images_cls(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, grayscale=grayscale, size=size)
                            print('[LFW][%d]Accuracy-Flip: %1.5f' % (batch+1, accuracy))
                        else:
                            if BACKBONE_NAME == 'Hybrid_ViT':
                                accuracy, std, xnorm, best_threshold, roc_curve = perform_val_color_images_hybrid_vit(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, grayscale=grayscale, size=size, use_scale=use_scale)
                            else:        
                                accuracy, std, xnorm, best_threshold, roc_curve = perform_val_color_images(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, grayscale=grayscale, size=size, use_scale=use_scale)

                            buffer_val(writer, name, accuracy, std, xnorm, best_threshold, roc_curve, batch + 1)
                            print('[%s][%d]XNorm: %1.5f' % (name, batch+1, xnorm))
                            print('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (name, batch+1, accuracy, std))
                            print('[%s][%d]Best-Threshold: %1.5f' % (name, batch+1, best_threshold))
                    acc.append(accuracy)
                    wandb.log({"acc LFW": accuracy})

                # save checkpoints per epoch
                if MULTI_GPU:
                    state_dict = BACKBONE.module.state_dict()
                else:
                    state_dict = BACKBONE.state_dict()

                torch.save(state_dict, os.path.join(WORK_PATH, "checkpoint.pth"))
                
                if accuracy > best_acc:
                    torch.save(state_dict, os.path.join(WORK_PATH, "best.pth"))
                    best_acc = accuracy
                print("highest_acc:", best_acc)

                # if need_save(acc, highest_acc):
                #     if MULTI_GPU:
                #         torch.save(BACKBONE.module.state_dict(), os.path.join(WORK_PATH, "Backbone_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(BACKBONE_NAME, epoch + 1, batch + 1, get_time())))
                #     else:
                #         torch.save(BACKBONE.state_dict(), os.path.join(WORK_PATH, "Backbone_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(BACKBONE_NAME, epoch + 1, batch + 1, get_time())))
                BACKBONE.train()  # set to training mode

            batch += 1 # batch index