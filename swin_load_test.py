import sys

from mmcv.runner import init_dist
from mmcv.utils import Config, DictAction, get_git_hash

from mmseg import __version__
from mmseg.apis import set_random_seed, train_segmentor
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import collect_env, get_root_logger
import torch


def main():
    model = get_model(args)
    print(model)
    print("Swin Model load Success")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Swin model load test')    
    
    parser.add_argument('--network', type=str,
                        default='swin_b',
                        const='swin_b',
                        nargs='?',
                        choices=['swin_s','swin_b','swin_t'],
                        help='swin_s, swin_b (base), swin_t (tiny)')

    parser.add_argument('--repo_root',type=str,default='')
    main()