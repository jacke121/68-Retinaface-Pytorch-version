
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import skimage
from skimage import io
from PIL import Image
import cv2
import torchvision
import eval_widerface
import torchvision_model
import os
import skimage
from dataloader import  ValDataset, Resizer, PadToSquare,ValDataset_CeleB
from torchvision import datasets, models, transforms
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
def pad_to_square(img, pad_value):
    _, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad

def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

def get_args():
    parser = argparse.ArgumentParser(description="Detect program for retinaface.")
    parser.add_argument('--image_path', type=str, default='WechatIMG10.jpeg', help='Path for image to detect')
    parser.add_argument('--model_path', type=str, help='Path for model',default="/versa/elvishelvis/RetinaYang/out/stage_4_68_full_model_epoch_61.pt")
    parser.add_argument('--save_path', type=str, default='./out', help='Path for result image')
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    args = parser.parse_args()

    return args

def main():
    args = get_args()

    # Create the model
    # if args.depth == 18:
    #     RetinaFace = model.resnet18(num_classes=2, pretrained=True)
    # elif args.depth == 34:
    #     RetinaFace = model.resnet34(num_classes=2, pretrained=True)
    # elif args.depth == 50:
    #     RetinaFace = model.resnet50(num_classes=2, pretrained=True)
    # elif args.depth == 101:
    #     RetinaFace = model.resnet101(num_classes=2, pretrained=True)
    # elif args.depth == 152:
    #     RetinaFace = model.resnet152(num_classes=2, pretrained=True)
    # else:
    #     raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    # Create torchvision model
    
    return_layers = {'layer2':1,'layer3':2,'layer4':3}
    model = torchvision_model.create_retinaface(return_layers)
    device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load trained model
    retina_dict = model.state_dict()
    pre_state_dict = torch.load('out/stage_5_68_full_model_epoch_121.pt',map_location='cpu')
    pretrained_dict = {k[7:]: v for k, v in pre_state_dict.items() if k[7:] in retina_dict}
    model.load_state_dict(pretrained_dict)
    # model.to(device)
    model.eval()
    model.cuda()
    import time
    
    video = cv2.VideoCapture(0)
    # Read image
    while True:

        ret, img_raw = video.read()

        img_raw = cv2.resize(img_raw, (480, 360))

        start = time.time()
        img=cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img)
        img = img.permute(2,0,1)
        input_img = img.float().unsqueeze(0)
        
        picked_boxes, picked_landmarks = eval_widerface.get_detections(input_img, model, score_threshold=0.9, iou_threshold=0.2)

        for j, boxes in enumerate(picked_boxes):
            if boxes is not None:
                for box,landmark in zip(boxes,picked_landmarks[j]):
                    cv2.rectangle(img_raw,(box[0],box[1]),(box[2],box[3]),(0,0,255),thickness=2)
                    for i in range(0,136,2):
                        cv2.circle(img_raw,(landmark[i],landmark[i+1]),radius=1,color=(0,0,255),thickness=2)
        cv2.imshow('RetinaFace-Pytorch',img_raw)
        print(time.time()-start)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
if __name__=='__main__':
    main()
