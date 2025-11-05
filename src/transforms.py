# makes images model-ready

import albumentations
from albumentations.pytorch import ToTensorV2

def get_transforms(split, img_size=512):
    if split == 'train':
        return albumentations.Compose([ # applies a series of transformations to the input image
            albumentations.Resize(img_size, img_size), # resizes the image to 512x512
            albumentations.HorizontalFlip(p=0.5), # there is a 0.5 probability of applying horizontal flip
            albumentations.Rotate(limit=15,p=0.5), # there is a 0.5 probability of applying rotation
            albumentations.RandomBrightnessContrast(p=0.5), # there is a 0.5 probability of applying brightness/contrast adjustment
            albumentations.Normalize(mean=(0,0,0), std=(1,1,1)), # scales the pixels to values between 0 and 1 dividing by 255
            ToTensorV2(), # coverts from Numpy float32 -> PyTorch float tensor [(h,w,c) -> (c,h,w)] where c is channels and h,w are height and width. For RGB images, c=3
        ])
    else:
        return albumentations.Compose([
            albumentations.Resize(img_size, img_size),
            albumentations.Normalize(mean=(0,0,0), std=(1,1,1)),
            ToTensorV2(),
        ])