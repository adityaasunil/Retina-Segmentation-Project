import sys,os 

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from src.dataset import RetinaVesselDataset
from src.models.unet import Unet

device = torch.device('mps')

vd = RetinaVesselDataset(None, 'val', 'splits/val.txt')
vl = torch.utils.data.DataLoader(vd, shuffle=False, batch_size=4)

model_path = os.path.join(PROJECT_ROOT,'outputs','ckpts','model_20251113_094305_53.pth')
model = Unet().to(device)
state = torch.load(model_path, map_location=device)
model.load_state_dict(state)


def evaluate_threshold(thresh, model, device):
    model.eval()
    dices=[]

    def dice_coeff(gt,pred,eps=1e-6):
        intersection = (gt*pred).sum(dim=(1,2))
        union = gt.sum(dim=(1,2)) + pred.sum(dim=(1,2))
        dice = (2 *intersection+ eps) / (union + eps)
        return dice

    with torch.no_grad():
        for i, (images, masks) in enumerate(vl):
            images = images.to(device) 
            masks = (masks>0.5).float().to(device) # binarize GT

            logits = model(images) # [B,1,H,W]
            probs = torch.sigmoid(logits[:,0,...]) # [B,H,W]
            preds = (probs>thresh).float()

            dice_batch = dice_coeff(masks.squeeze(1), preds)
            dices.append(dice_batch)

    dices = torch.cat(dices, dim=0)
    return dices.mean().item()

thresholds = [0.3,0.35,0.4,0.45,0.5,0.55,0.6]
best_t = None
best_dice = -1

for t in thresholds:
    mean_d = evaluate_threshold(t, model, device)
    print("Threshold {:.2f} -> mean Dice {:.4f}".format(t,mean_d))
    if mean_d > best_dice:
        best_dice = mean_d
        best_t = t 

print("-> Best threshold: {:.2f} with Dice: {:.4f}".format(best_t,best_dice))