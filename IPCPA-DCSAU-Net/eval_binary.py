# 二分类
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from data_loading import binary_class
import albumentations as A
from albumentations.pytorch import ToTensor
from pytorch_lightning.metrics import Accuracy, Precision, Recall, F1
import argparse
import time
import pandas as pd
import cv2
import os
from skimage import io, transform
from PIL import Image


class IoU(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoU, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return IoU

class Dice(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(Dice, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return dice

def get_transform():
   return A.Compose(
       [
        A.Resize(64, 64),  ########################################################
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensor()
        ])
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='E:/resu_net/datasets/DDTI/2_preprocessed_data/stage2/',type=str, help='the path of dataset')  ###########################################
    parser.add_argument('--csvfile', default='src/test_train_data.csv',type=str, help='two columns [image_id,category(train/test)]')
    parser.add_argument('--model',default='save_models/epoch_last.pth', type=str, help='the path of model')   #########################################
    parser.add_argument('--debug',default=True, type=bool, help='plot mask')
    args = parser.parse_args()
    
    os.makedirs('debug/',exist_ok=True)
    
    df = pd.read_csv(args.csvfile)
    df = df[df.category=='test']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_files = list(df.image_id)
    test_dataset = binary_class(args.dataset,test_files, get_transform())
    model = torch.load(args.model)

    model = model.cuda()
    
    acc_eval = Accuracy()
    pre_eval = Precision()
    dice_eval = Dice()
    recall_eval = Recall()
    f1_eval = F1(2)
    iou_eval = IoU()
    iou_score = []
    acc_score = []
    pre_score = []
    recall_score = []
    f1_score = []
    dice_score = []
    time_cost = []
    
    since = time.time()
    
    for image_id in test_files:
        img = cv2.imread(f'E:/resu_net/datasets/DDTI/2_preprocessed_data/stage2/images/{image_id}')     #在字符串中插入变量的值，可在前引号前加上字母f，再将要插入的变量放在花括号内
        print('img:',image_id)
        img = cv2.resize(img, ((64,64)))   ################################
        img_id = list(image_id.split('.'))[0]
        cv2.imwrite(f'debug/{img_id}.png',img)
    
    with torch.no_grad():
        for img, mask, img_id in test_dataset:
            print(img.shape)
            img = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False).cuda()           
            mask = Variable(torch.unsqueeze(mask, dim=0).float(), requires_grad=False).cuda()
            torch.cuda.synchronize()
            start = time.time()
            pred = model(img)    # 预测输出
            print('pred.shape:',pred.shape)
            torch.cuda.synchronize()   # 用于同步CPU和GPU之间的计算
            end = time.time()
            time_cost.append(end-start)

            pred = torch.sigmoid(pred)   # 转化为0-1之间

            pred[pred >= 0.5] = 1   # 转化为0，1矩阵
            pred[pred < 0.5] = 0
            

            pred_draw = pred.clone().detach()
            mask_draw = mask.clone().detach()
            
            
            if args.debug:
                img_id = list(img_id.split('.'))[0]
                img_numpy = pred_draw.cpu().detach().numpy()[0][0]
                img_numpy[img_numpy==1] = 255 
                cv2.imwrite(f'debug/{img_id}_pred.png',img_numpy)
                
                mask_numpy = mask_draw.cpu().detach().numpy()[0][0]
                mask_numpy[mask_numpy==1] = 255
                cv2.imwrite(f'debug/{img_id}_gt.png',mask_numpy)
            iouscore = iou_eval(pred,mask)  ##
            dicescore = dice_eval(pred,mask)
            pred = pred.view(-1)
            mask = mask.view(-1)
     
            accscore = acc_eval(pred.cpu(),mask.cpu())  ############################
            prescore = pre_eval(pred.cpu(),mask.cpu())
            recallscore = recall_eval(pred.cpu(),mask.cpu())
            f1score = f1_eval(pred.cpu(),mask.cpu())
            iou_score.append(iouscore.cpu().detach().numpy())
            dice_score.append(dicescore.cpu().detach().numpy())
            acc_score.append(accscore.cpu().detach().numpy())
            pre_score.append(prescore.cpu().detach().numpy())
            recall_score.append(recallscore.cpu().detach().numpy())
            f1_score.append(f1score.cpu().detach().numpy())
            torch.cuda.empty_cache()
            
    time_elapsed = time.time() - since
    
    print('Evaluation complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('FPS: {:.2f}'.format(1.0/(sum(time_cost)/len(time_cost))))
    print('mean IoU:',round(np.mean(iou_score),4),round(np.std(iou_score),4))
    print('mean accuracy:',round(np.mean(acc_score),4),round(np.std(acc_score),4))
    print('mean precsion:',round(np.mean(pre_score),4),round(np.std(pre_score),4))
    print('mean recall:',round(np.mean(recall_score),4),round(np.std(recall_score),4))
    print('mean F1-score:',round(np.mean(f1_score),4),round(np.std(f1_score),4))
