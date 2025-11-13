import os, sys, torch, torchvision 
from datetime import datetime 

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset import RetinaVesselDataset
from src.models.unet import Unet

# --- CONFIG --- 
CKPT_PATH = os.path.join(PROJECT_ROOT, 'outputs', 'ckpts' ,'model_20251113_094305_53.pth')
BATCH_SIZE = 4
SAVE_DIR = os.path.join(PROJECT_ROOT, "outputs", "preds", datetime.now().strftime('%Y%m%d_%H%M%S'))
THRESH = 0.55

os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device('mps')

ds = RetinaVesselDataset(None, split='test', split_file='splits/test.txt')

print('Loaded Test instances: {}'.format(len(ds)))

dl = torch.utils.data.DataLoader(ds, shuffle=False, batch_size=BATCH_SIZE)

model = Unet().to(device)
state = torch.load(CKPT_PATH, map_location=device)
model.load_state_dict(state)
model.eval()

def dice_coeff(y_true, y_pred, smooth=1e-6): # y_true, y_pred -> [B,H,W]
    intersection = (y_true*y_pred).sum(dim=(1,2)) # we only need [H,W]
    sum  = y_true.sum(dim=(1,2)) + y_pred.sum(dim=(1,2))
    dice = (2 * intersection + smooth)/ (sum + smooth)
    return dice

def get_binary_pred(logits: torch.tensor, threshold: float = 0.6):
    '''Return (probs, preds) as [B,H,W] floats in [0,1] and {0,1}.
    Supports logits of shape [B,1,H,W] (BCE) or [B,2,H,W] (CE with class-1 as vessel)
    '''
    if logits.ndim != 4:
        raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)}")
    if logits.shape[1] == 1:
        probs = torch.sigmoid(logits[:, 0, ...])
    elif logits.shape[1] == 2:
        probs = torch.softmax(logits, dim=1)[:,1,...]
    else:
        raise RuntimeError(f"Unsupported out_channels={logits.shape[1]} expect 1 or 2")
    preds = (probs > threshold).float()
    return probs, preds
    
all_dice = []

with torch.no_grad():
    for i, (images, masks) in enumerate(dl):
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)

        probs, preds = get_binary_pred(logits, threshold=THRESH)
        preds = preds.to(device)
        
        gt = (masks.squeeze(1)>0.5).float()

        all_dice.append(dice_coeff(gt, preds))

        imgs_disp = images.detach().cpu().clamp(0,1)
        if imgs_disp.shape[1] == 1:
            imgs_disp = imgs_disp.repeat(1,3,1,1)

        gt_disp = gt.detach().cpu().unsqueeze(1).repeat(1,3,1,1)
        preds_disp = preds.detach().cpu().unsqueeze(1).repeat(1,3,1,1)

        grid = torchvision.utils.make_grid(
            torch.cat([imgs_disp,gt_disp,preds_disp], dim=0),
            nrow=images.size(0),pad_value=0.2
        )
        torchvision.utils.save_image(
            grid,
            os.path.join(SAVE_DIR, f"batch{i+1}.png")
        )

all_dice = torch.cat(all_dice, dim=0)
avg_dice = all_dice.mean().item()
print(f'Saved prediction to : {SAVE_DIR}')
print(f"Mean Dice on {len(ds)} images: {avg_dice}")
