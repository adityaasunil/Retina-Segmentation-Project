import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch 
import torchvision 
from src.dataset import RetinaVesselDataset
from src.models.unet import Unet
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime 
import matplotlib.pyplot as plt 
import numpy as np

device = torch.device('mps')

batch_size = 4
epoch = 100
learning_rate = 1e-3


training_set = RetinaVesselDataset(None, "train", "splits/train.txt")
training_set_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True)
validation_data_set = RetinaVesselDataset(None, "val", "splits/val.txt")
validation_set_loader = torch.utils.data.DataLoader(validation_data_set, batch_size=batch_size, shuffle=False)

print('Training set has {} instances'.format(len(training_set)))
print('Validation set has {} instances'.format(len(validation_data_set)))

# ------- Visualizing the data -----------

# # Helper function for inline image display
# def matplotlib_imshow(img, one_channel=False):
#     if one_channel:
#         img = img.mean(dim=0)
#     img = img/ 2 + 0.5 # unnormalize
#     npimg = img.numpy()
#     if one_channel:
#         plt.imshow(npimg, cmap='grey')
#     else:
#         plt.imshow(np.transpose(npimg, (1,2,0)))

# dataiter = iter(training_set_loader)
# images, labels = next(dataiter)

# # Create a grid from the images and masks, then show them stacked
# img_grid = torchvision.utils.make_grid(images)
# mask_grid = torchvision.utils.make_grid(labels)

# plt.figure(figsize=(8, 8))
# plt.subplot(2, 1, 1)
# plt.title("Input images")
# matplotlib_imshow(img_grid, one_channel=False)

# plt.subplot(2, 1, 2)
# plt.title("Corresponding masks")
# matplotlib_imshow(mask_grid, one_channel=True)

# plt.tight_layout()
# plt.show()

pos_weight = torch.tensor([3.0], device=device)
bce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

def dice_loss(logits, target, eps=1e-6):
    logits  = logits.squeeze(1) #[B,1,H,W] -> [B,H,W]
    target = target.squeeze(1) 
    probs = torch.sigmoid(logits)
    num = 2 * (target * probs).sum(dim=(1,2)) + eps
    den = probs.sum(dim=(1,2)) + target.sum(dim=(1,2)) + eps
    dice = num / den
    return 1 - dice.mean()

# ----- Sanity check for loss fn ------

# dummy_outputs = torch.rand(4,16)
# dummy_masks = torch.tensor([1,5,4,7])

# print(dummy_outputs)
# print(dummy_masks)

# loss = loss_fn(dummy_outputs, dummy_masks)
# print('Total loss for this batch {}'.format(loss.item()))


# ---- Defining the optimizer ------
model = Unet().to(device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer,
    mode='min', # trying to minimize validation loss
    factor=0.5,
    patience=5,
)

def get_loss_weights(epoch_index, total_epochs):
    """
    Returns (alpha, beta) for BCE and Dice
    Start: more BCE, End: more Dice
    """

    # progress from 0->100
    progress = epoch_index / max(1, total_epochs-1)

    #linearly go from (alpha=0.8, beta=0.2) to (alpha=0.4, beta=0.6)
    alpha_start , alpha_end = 0.8, 0.2
    beta_start, beta_end = 0.2,0.8

    alpha = alpha_start + (alpha_end - alpha_start) * progress
    beta = beta_start + (beta_end - beta_start) * progress

    return alpha, beta


# ---- Training one epoch -----

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0
    last_loss = None

    alpha, beta = get_loss_weights(epoch_index, epoch)
    print("-> Using alpha={:.3f} and beta={:.3f}".format(alpha,beta))

    for i, (images, masks) in enumerate(training_set_loader):

        images = images.to(device)
        masks = (masks>0.5).float()
        masks = masks.float().to(device)

        # zeroing gradients after every batch
        optimizer.zero_grad()

        # make predictions for this batch
        outputs = model(images)

        # compute the loss and its gradients
        loss = alpha * bce_loss(outputs, masks) + beta * dice_loss(outputs, masks)
        loss.backward()

        # adjust learning weights
        optimizer.step()

        running_loss += loss.item()
        if (i+1) % 4 == 0:
            last_loss = running_loss / 4.0
            tb_x = epoch_index * len(training_set_loader) + (i + 1)
            tb_writer.add_scalar('Loss/Train', last_loss, tb_x)
            running_loss = 0.0

    return float(last_loss) if last_loss is not None else 0.0

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_dir = os.path.join(PROJECT_ROOT, 'outputs', 'logs', f'retina_seg_model_{timestamp}')
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir)
epoch_number = 0

best_vloss = 1_000_000.

for e in range(epoch):
    print('-> EPOCH {}/{}'.format(e+1,epoch))
    
    current_lr = optimizer.param_groups[0]['lr']
    print('-> Current LR: {:.6e}'.format(current_lr))

    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)

    running_vloss = 0.0
    model.eval()

    dice=[]

    with torch.no_grad():
        for i, (images,masks) in enumerate(validation_set_loader):
            images = images.to(device)
            val_outputs = model(images)
            masks = (masks>0.5).float().to(device)
            val_loss = 0.5*bce_loss(val_outputs, masks) + 0.5*dice_loss(val_outputs, masks)
            dice.append(dice_loss(val_outputs,masks))
            running_vloss += val_loss.item()

    dice = torch.stack(dice)
    avg_dice = dice.mean().item()
    avg_vloss = running_vloss/ len(validation_set_loader)
    print('-> Train loss: {:.6f} Validation Loss: {:.6f} Validation Dice Loss: {:.6f}'.format(avg_loss, avg_vloss, avg_dice))

    scheduler.step(avg_vloss)

    writer.add_scalars('Training vs Validation Loss',
                       {'Training' : float(avg_loss),  'Validation' : float(avg_vloss)},
                       epoch_number+1)
    writer.flush()

    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        Checkpoint_path = os.path.join(PROJECT_ROOT, 'outputs', 'ckpts')
        model_path = os.path.join(Checkpoint_path,'model_{}_{}.pth'.format(timestamp, epoch_number))
        torch.save(model.state_dict(), model_path)

    epoch_number += 1
