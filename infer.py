import torch
import torch.nn as nn
from torch.nn import functional as F
from pycocotools.coco import COCO


import pandas as pd
import torchvision
import torchvision.transforms as transforms
import albumentations as A
from utils import *
from dataloader import *

def collate_fn(batch):
    return tuple(zip(*batch))

def infer(args):
    dataset_path = 'input/data'
    test_path = dataset_path + '/test.json'

    test_transform = A.Compose([
                          A.Resize(args.infer_size , args.infer_size ),
    A.Normalize (mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225],max_pixel_value=1.0),
    ToTensorV2(),
                          ])
    transform = A.Compose([A.Resize(256, 256)])

    test_dataset = CustomDataLoader(data_dir=test_path, mode='test', transform=test_transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=16,
                                          num_workers=4,
                                          collate_fn=collate_fn)
    model = get_model(args,train=False)

    device = 'cuda'
    load_model(model,device,saved_dir=args.weight_dir,file_name = args.weight_file)
    model.to(device)
    model.eval()

    size= 256
    file_name_list = []
    preds_array = np.empty((0, size*size), dtype=np.long)

    # fake metas
    img_metas =[[{
        'img_shape': (args.infer_size , args.infer_size , 3),
        'ori_shape': (args.infer_size , args.infer_size , 3),
        'pad_shape': (args.infer_size , args.infer_size , 3),
        'filename': '<demo>.png',
        'scale_factor': 1.0,
        'flip': False,
                'flip_direction': 'horizontal'
            }]]

    with torch.no_grad():
        for step, (imgs, image_infos) in enumerate(test_loader):
            in_imgs = torch.stack(imgs).to(device)
            
            outs = model([in_imgs],img_metas,return_loss=False)
            oms = torch.argmax(outs.squeeze(), dim=1).detach().cpu().numpy()
            
            # resize (256 x 256)
            if args.infer_size != 256:
                temp_mask = []
                for img, mask in zip(np.stack(imgs), oms):
                    transformed = transform(image=img, mask=mask)
                    mask = transformed['mask']
                    temp_mask.append(mask)

                oms = np.array(temp_mask)
            
            oms = oms.reshape([oms.shape[0], size*size]).astype(int)
            preds_array = np.vstack((preds_array, oms))
            
            file_name_list.append([i['file_name'] for i in image_infos])
    print("End prediction.")
    file_names = [y for x in file_name_list for y in x]

    submission = pd.read_csv('code/submission/sample_submission.csv', index_col=None)

    # PredictionString 대입
    for file_name, string in zip(file_names, preds_array):
        submission = submission.append({"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in string.tolist())}, 
                                    ignore_index=True)

    # submission.csv로 저장
    submission.to_csv(f"code/submission/{args.weight_file}.csv", index=False)


if __name__=='__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('--weight_dir', type=str,default='weight',help='Path of weight file')
    parser.add_argument('--infer_size', type=int,default=256)
    parser.add_argument('--weight_file',type=str,default='',help ='Name of weight file')
    parser.add_argument('--network', type=str,
                        default='swin_b',
                        const='swin_b',
                        nargs='?',
                        choices=['swin_s','swin_b','swin_t'],
                        help='swin_s, swin_b (base), swin_t (tiny)')
    
    args = parser.parse_args()
    
    infer(args)