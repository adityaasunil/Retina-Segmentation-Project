from dataset import RetinaVesselDataset
import matplotlib.pyplot as plt
import numpy as np 
import os


'''
    This is basically to check whether all the files in the image-train, and mask-train overlay
    each other perfectly.
'''


def load_example(fname):
    dataset= RetinaVesselDataset(None, "test", "splits/test.txt")
    img_tensor, mask_tensor, file_name = dataset[fname] # tensor is in the form (C,H,W)
    def tensortonumpy(img):
        return img.permute(1,2,0).cpu().numpy() # converts from (C,H,W) -> (H,W,C) , basically changing the indexes
                                                             #. (0,1,2) -> (1,2,0) => because matplotlib expects values in the order (H,W,C)
    npimg = tensortonumpy(img_tensor)
    npmask = tensortonumpy(mask_tensor)
    return npimg, npmask

img,mask = load_example(2)
plt.imshow(img)
plt.imshow(mask,cmap='jet',alpha=0.5)
plt.show()