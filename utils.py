import os

import numpy as np
import torch
import copy
from mmcv.runner import init_dist, load_checkpoint
from mmcv.utils import Config, DictAction, get_git_hash

from mmseg import __version__
from mmseg.apis import set_random_seed, train_segmentor, init_segmentor
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import collect_env, get_root_logger

import segmentation_models_pytorch as smp

def save_model(model, saved_dir="model", file_name="default.pt"):
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)
    check_point = {'model' : model.state_dict()}
    path = os.path.join(saved_dir, file_name)
    torch.save(check_point, path)


def load_model(model, device, saved_dir="model", file_name="default.pt"):
    path = os.path.join(saved_dir, file_name)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(state_dict=checkpoint['model'])
    print("load success")


def load_swin(cfg,ck_path):
    # make temp_cfg
    temp_cfg = copy.deepcopy(cfg)
    temp_cfg.model.decode_head.num_classes=150
    temp_cfg.model.auxiliary_head.num_classes=150

    # make original swin
    pre_trained = build_segmentor(temp_cfg.model,train_cfg=None,
        test_cfg=None)
    load_checkpoint(pre_trained,ck_path, map_location='cpu')

    model = build_segmentor(cfg.model,train_cfg=None,
        test_cfg=None)

    # make pretrained state dict
    pretrained_dict = pre_trained.backbone.state_dict()
    model_dict = model.backbone.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict) 
    del pre_trained
    torch.cuda.empty_cache()
    model.backbone.load_state_dict(model_dict)
    return model


def make_swin_model(cfg):
    return build_segmentor(cfg.model,train_cfg=None,test_cfg=None)


def get_model(args,classes=12,train=True):
    repo_root = args.repo_root
    
    if args.network[-1] == 's':
        cfg = Config.fromfile(os.path.join(repo_root,'swin/configs/swin/upernet_swin_small_patch4_window7_512x512_160k_ade20k.py'))
        ck_path = os.path.join(repo_root,'pretrained/upernet_swin_small_patch4_window7_512x512.pth')
    elif args.network[-1] == 'b':
        cfg = Config.fromfile(os.path.join(repo_root,'swin/configs/swin/upernet_swin_base_patch4_window7_512x512_160k_ade20k.py'))
        ck_path = os.path.join(repo_root,'pretrained/upernet_swin_base_patch4_window7_512x512.pth')
    else:
        cfg = Config.fromfile(os.path.join(repo_root,'swin/configs/swin/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k.py'))
        ck_path = os.path.join(repo_root,'pretrained/upernet_swin_tiny_patch4_window7_512x512.pth')        
    
    if train:
        return load_swin(cfg,ck_path)
    else:
        return make_swin_model(cfg)
