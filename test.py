from models.vit_model_face import ViT_face_model, Hybrid_ViT
from models.resnet import resnet_face18
import torch
import sys
from models import ViT_face
import argparse
from util.utils import get_val_data, perform_val_color_images_cls, perform_val_resnet_color_images, perform_val_color_images, perform_val_color_images_hybrid_vit, AverageMeter

def main(args):
    GPU_ID = [0]
    device = torch.device('cuda:%d' % GPU_ID[0])
    torch.backends.cudnn.benchmark = True
    NUM_CLASS =10575 # for LFW # 93431 for casia
    depth_vit = 8
    heads = 1
    channels = 1
    size = 128 #112 
    out_dim = 512 
    isHybrid = True
    use_scale = True
    grayscale = True

    use_face_loss = True
    use_cls = False
    HEAD_NAME = 'ArcFace'
    name = args.name #'talfw' #'mlfw' # # # #'glfw'  # #

    if isHybrid == False:
        model = ViT_face_model(loss_type='ArcFace',
                            GPU_ID=['0'],
                            num_class=NUM_CLASS,
                            use_cls=use_cls,
                            use_face_loss=use_face_loss,
                            no_face_model=False,
                            image_size=112,
                            patch_size=8,
                            ac_patch_size=12,
                            pad = 4,
                            dim=512,
                            depth=depth_vit,
                            heads=heads,
                            mlp_dim=2048,
                            dropout=0.1,
                            emb_dropout=0.1,
                            out_dim=out_dim,
                            singleMLP=False,
                            remove_sep=False)
    else:
        model = Hybrid_ViT(loss_type=HEAD_NAME,
                            GPU_ID=GPU_ID,
                            num_class=NUM_CLASS,
                            channels=channels,
                            image_size=size,
                            patch_size=8,
                            ac_patch_size=12,
                            pad = 4,
                            dim=512, #256,
                            depth=depth_vit,
                            heads=heads,
                            mlp_dim=2048,
                            dropout=0.1,
                            emb_dropout=0.1,
                            out_dim=out_dim, 
                            remove_pos=False)

    # model = ViT_face(loss_type = HEAD_NAME,
    #                 GPU_ID = '0',
    #                 num_class = NUM_CLASS,
    #                 image_size=size,
    #                 patch_size=8,
    #                 dim=512,
    #                 depth=depth_vit,
    #                 heads=heads,
    #                 mlp_dim=2048,
    #                 dropout=0.1,
    #                 emb_dropout=0.1)
    
    # model_path = 'results/ViT-face_webface_arcface_resnet18_gray_depth_1_3e4/Backbone_VIT_face_Epoch_3_Batch_500_Time_2022-02-26-19-24_checkpoint.pth'
    # model_path = 'results/ViT-face_webface_2m_depth_2_resnet18_gray_arcface_1e5/Backbone_VIT_face_Epoch_47_Batch_2000_Time_2022-03-04-00-11_checkpoint.pth'
    
    # model_path = 'results/ViT-face_webface_arcface_resnet18_gray_depth_1_head_2_lr_1e6/best.pth'
    # model_path = 'results/ViT-face_webface_arcface_resnet18_gray_depth_1_head_2_lr_1e6/best.pth'
    # model_path = 'results/ViT-face_webface_arcface_resnet18_gray_depth_1_head_2_1e7/best.pth'
    # model_path = 'results/ViT-face_webface_arcface_resnet18_gray_depth_1_head_2_lr_1e-5_fc1024_dropout_0_LFW/best.pth'
    # model_path = 'results/ViT-face_msceleb_arcface_resnet18_gray_depth_1_head_2_lr_1e-5_fc512_dropout_0_LFW/best.pth'
    # model_path = 'results/ViT-face_webface_2m_depth_1_head_6_resnet18_lr1e5/best.pth'
    # model_path = 'results/ViT-face_webface_2m_depth_1_head_4_resnet18_gray_arcface_1e5/best.pth'
    # model_path = 'results/ViT-face_webface_2m_depth_1_head_8_resnet18_1024fc_lr1e5/best.pth'

    if use_cls:
        # model_path = 'results/ViT-P8S8_2-image_webface_2m_arcface_s1_depth_1_head_1_use_cls/best.pth'
        # model_path = 'results/ViT-P8S8_2-image_webface_2m_arcface_resnet18_s1_depth_1_head_1_use_cls/best.pth'
        # model_path = 'results/ViT-P8S8_2-image_webface_2m_arcface_s1_depth_1_head_2_use_cls/best.pth'
        model_path = 'results/ViT-P8S8_2-image_webface_2m_arcface_s1_depth_2_head_1_use_cls/best.pth'
    else:
        if isHybrid:
            model_path = 'results/ViT-P8S8_2-image_webface_2m_arcface_resnet18_s1_depth_{}_head_{}_hybrid/best.pth'.format(depth_vit, heads)
        else:
            model_path = 'results/ViT-P8S8_2-image_webface_2m_arcface_resnet18_s1_depth_{}_head_{}_LFW_lr1e5/best.pth'.format(depth_vit, heads) 

    print('model path: {}'.format(model_path))
    # model_path = 'results/ViT-face_webface_arcface_resnet18_gray_depth_1_head_1_lr_1e-5_fc1024_dropout_0_LFW/best.pth'
    # face_model_moopath =  'pretrained/resnet18_110.pth'
    # state_dict = torch.load(face_model_path, map_location=torch.device('cpu'))
    facemodel = resnet_face18(False, use_reduce_pool=False, grayscale=True)
    
    # model_path = 'results/ViT-P8S8_ms1m_arcface_s1/best.pth'

    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     name = k[7:] # remove module.
    #     new_state_dict[name] = v
    # facemodel.load_state_dict(new_state_dict)

    
    model.face_model = facemodel
    model.load_state_dict(torch.load(model_path))

    BATCH_SIZE = 64
    EMBEDDING_SIZE = out_dim
    # size = 128
    MULTI_GPU = False
    EVAL_PATH = './eval/'
    TARGET = ['mlfw']

    # for ver in vers:
    #     name, data_set, issame = ver
    # vers = get_val_data(EVAL_PATH, TARGET)

    
    print('Process [{}] dataset, model depth={}, heads={}, emb={}'.format(name, depth_vit, heads, EMBEDDING_SIZE))
    # name, data_set, issame = vers[0]
    # accuracy, std, xnorm, best_threshold, roc_curve = perform_val(MULTI_GPU, device, EMBEDDING_SIZE, BATCH_SIZE, model, data_set, issame)
    
    # accuracy, std, xnorm, best_threshold, roc_curve = perform_val_resnet_color_images(MULTI_GPU, device, EMBEDDING_SIZE, BATCH_SIZE, model.face_model, grayscale=True, size=size, use_scale=use_scale, target=name)
    if use_cls:
        accuracy = perform_val_color_images_cls(MULTI_GPU, device, EMBEDDING_SIZE, BATCH_SIZE, model, grayscale=grayscale, size=size, target=name)
        print('[%s]Accuracy-Flip: %1.5f' % (name, accuracy))
    else:
        if isHybrid:
            accuracy, std, xnorm, best_threshold, roc_curve = perform_val_color_images_hybrid_vit(MULTI_GPU, device, EMBEDDING_SIZE, BATCH_SIZE, model, grayscale=grayscale, size=size, use_scale=use_scale, target=name)
        else:
            accuracy, std, xnorm, best_threshold, roc_curve = perform_val_color_images(MULTI_GPU, device, EMBEDDING_SIZE, BATCH_SIZE, model, grayscale=grayscale, size=size, use_scale=use_scale, target=name)
    
    # perform_val_color_images

        print('[%s]XNorm: %1.5f' % (name, xnorm))
        print('[%s]Accuracy-Flip: %1.5f+-%1.5f' % (name, accuracy, std))
        print('[%s]Best-Threshold: %1.5f' % (name, best_threshold))

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='', help='training set directory')
    parser.add_argument('--name', default='lfw', help='test set')
    parser.add_argument('--network', default='VITs',
                        help='training set directory')
    parser.add_argument('--target', default='lfw,talfw,sllfw,calfw,cplfw,cfp_fp,agedb_30',
                        help='')
    parser.add_argument('--batch_size', type=int, help='', default=20)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))