import os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


import cv2 
import numpy as np 
import torch as t 
import torch.utils.data.dataset as dataset 
from src.transforms import get_transforms


class RetinaVesselDataset(dataset.Dataset):
    def __init__(self,root,split,split_file,img_size=512):

        self.root = root 
        self.split = split 
        self.split_file = split_file
        self.img_size = img_size

        if (root == None):
            self.root = ROOT

        self.split_file = split_file if os.path.isabs(split_file) else os.path.join(ROOT, split_file)

        if split == "train":
            img_dir = "data/image/image-train"
            mask_dir = "data/image/mask-train"
        elif split=='val':
            img_dir = "data/image/image-val"
            mask_dir = "data/image/mask-val"
        else:
            img_dir = "data/image/image-test"
            mask_dir = "data/image/mask-test"

        self.img_dir = img_dir
        self.mask_dir = mask_dir

        with open(self.split_file,'r') as f: #opens the split file and reads all the lines and strips each line to remove "\n" and saves to self.files
            self.files = [line.strip() for line in f.readlines()]

        self.tfs = get_transforms(split,img_size) # gets the transformations for the given split

    def __len__(self):
        return len(self.files) # returns the number of files in the split
        
    def __getitem__(self,idx):
        fname = self.files[idx]
        img_path = os.path.join(self.root,self.img_dir,fname)
        mask_path = os.path.join(self.root,self.mask_dir,fname)

            
        img = cv2.imread(img_path) # reads the image using OpenCV
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) # converts BGR to RGB
        mask = cv2.imread(mask_path,0) # reads the mask in grayscale mode

        mask = ((mask.astype(np.float32))/255.0)# scales the mask to [0,1]
        mask  = (mask>0.5).astype(np.float32) # binarize before any transforms

        transformed = self.tfs(image=img,mask=mask)
        img_tensor = transformed['image']
        mask_tensor = transformed['mask']

        if isinstance(mask_tensor,np.ndarray): # asking whether mask_tensor is a numpy array
            mask_tensor = (mask_tensor>0.5).astype(np.float32) #normalizing values greater than 0.5 as 1, and less than that to 0, and storing them as a float
            mask_tensor = t.from_numpy(mask_tensor) # converting to Pytorch tensor
        else:
            mask_tensor = (mask_tensor>0.5).float()


        if mask_tensor.ndim == 2: #checks the number of dimensions in that array
            mask_tensor = mask_tensor.unsqueeze(0) #since, pytorch requires 3 dimensions, we add a fake dimension at the 0th index of that vector
        mask_tensor = mask_tensor.float()


        return img_tensor, mask_tensor
