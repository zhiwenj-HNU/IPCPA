import os
from skimage import io, transform, color,img_as_ubyte
import numpy as np
from torch.utils.data import Dataset
import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensor

def Normalization():
   return A.Compose(
       [
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensor()
        ])


#Dataset Loader
class multi_classes(Dataset):
        def __init__(self,path,data, transform=None):
            self.path = path
            self.folders = data
            self.transforms = transform
            self.normalization = Normalization()
        
        def __len__(self):
            return len(self.folders)
              
        
        def __getitem__(self,idx):
            image_folder = os.path.join(self.path,str(self.folders[idx]),'images/')
            mask_folder = os.path.join(self.path,str(self.folders[idx]),'masks/')
            print('os.listdir(image_folder)[0]:',image_folder)
            image_path = os.path.join(image_folder,os.listdir(image_folder)[0])
            
            image_id = self.folders[idx]
            img = io.imread(image_path)[:,:,:3].astype('float32')         
            mask = self.get_mask(mask_folder, 256, 256)
   
            augmented = self.transforms(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
            
            normalized = self.normalization(image=img, mask=mask)
            img_nl = normalized['image']
            mask_nl = normalized['mask']
            mask_nl = np.squeeze(mask_nl)
            
            mask = img_as_ubyte(mask) 
            mask = np.squeeze(mask)
            mask[(mask > 0) & (mask < mask.max())] = 1
            mask[mask == mask.max()] = 2
            mask = torch.from_numpy(mask)
            mask = torch.squeeze(mask)
            mask = torch.nn.functional.one_hot(mask.to(torch.int64),3)
            mask = mask.permute(2, 0, 1)
            return (img_nl,mask,mask_nl,image_id) 


        def get_mask(self,mask_folder,IMG_HEIGHT, IMG_WIDTH):
            mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
            for mask_ in os.listdir(mask_folder):
                    mask_ = io.imread(os.path.join(mask_folder,mask_), as_gray=True)
                    mask_ = transform.resize(mask_, (IMG_HEIGHT, IMG_WIDTH))
                    mask_ = np.expand_dims(mask_,axis=-1)
                    mask = np.maximum(mask, mask_)

            return mask

class binary_class(Dataset):
        def __init__(self,path,data, transform=None):
            self.path = path
            self.folders = data
            self.transforms = transform
        
        def __len__(self):
            return len(self.folders)
              
        
        def __getitem__(self,idx):
            image_path = os.path.join(self.path,'images/',self.folders[idx])
            mask_path = os.path.join(self.path,'masks/',self.folders[idx])
            # mask_a,mask_b=mask_path.split('.')
            # mask_path=mask_a+'_mask.'+mask_b
            #print('image_path:',self.folders[idx])
            #img = io.imread(image_path)[:,:,:3].astype('float32')
            img = io.imread(image_path).astype('float32')
            #print('imamge_path:',image_path)  #####################################################
            #img = io.imread(image_path)
            #print('img:',type(img))    ################################################################

            #print('mask_path:',mask_path)

            mask = io.imread(mask_path)
            #print('mask.shape:',mask.shape)  ######################################################
            # if mask.shape[-1]!=64:
            #     mask = io.imread(mask_path)[:,:,0]
            # for i in range(mask.shape[0]):
            #     print('mask:',mask[i])
            #mask=(mask*255).astype(int)
            #print('mask.shape:',mask)  ######################################################
            #print('mask：',mask.size)  ##########################
            image_id = self.folders[idx]
            
            augmented = self.transforms(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
            
            return (img,mask,image_id)
        
class binary_class2(Dataset):
        def __init__(self,path,data, transform=None):
            self.path = path
            self.folders = data
            self.transforms = transform
        
        def __len__(self):
            return len(self.folders)
              
        
        def __getitem__(self,idx):
            image_path = os.path.join(self.path,self.folders[idx],'images/',self.folders[idx])
            mask_path = os.path.join(self.path,self.folders[idx],'masks/',self.folders[idx])
            image_id = self.folders[idx]
            img = io.imread(f'{image_path}.tif')[:,:,:3].astype('float32') ##################
            mask = io.imread(f'{mask_path}.tif', as_gray=True)  ###############################

            augmented = self.transforms(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
   
            return (img,mask,image_id)
        
