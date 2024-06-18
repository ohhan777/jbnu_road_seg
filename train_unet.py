import os
import argparse
import torch
from utils.kari_road_dataset import KariRoadDataset
from utils.metrics import ConfusionMatrix
from utils.loss import ce_loss
from models.unet import Unet
import torch.optim as optim
import time
import wandb
from pathlib import Path
from torchvision import models
from utils.plots import plot_image

def train(opt):
    epochs = opt.epochs
    batch_size = opt.batch_size
    name = opt.name
    # wandb settings
    wandb.init(id=opt.name, resume='allow')
    wandb.config.update(opt)

    # Train dataset
    train_dataset = KariRoadDataset('./data/kari-road', train=True)
    # Train dataloader
    num_workers = min([os.cpu_count(), batch_size, 16])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=False)

    # Validation dataset
    val_dataset = KariRoadDataset('./data/kari-road', train=False)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=False)
    
    # Network model
    num_classes = 10 # background + 1 classes
    # model = models.segmentation.deeplabv3_resnet101(num_classes=num_classes)
    model = Unet(backbone='convnext_small', in_channels=3, num_classes=num_classes)  
    
    # GPU-support
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:   # multi-GPU
        model = torch.nn.DataParallel(model)
    model.to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
      
    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # loading a weight file (if exists)
    weight_file = Path('weights')/(name + '.pth')
    best_accuracy = 0.0
    start_epoch, end_epoch = (0, epochs)
    if os.path.exists(weight_file):
        checkpoint = torch.load(weight_file)
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch'] + 1
        best_accuracy = checkpoint['best_accuracy']
        print('resumed from epoch %d' % start_epoch)

    (img, label_img, img_file) = train_dataset[0]
    
    confusion_matrix = ConfusionMatrix(num_classes)

    # training/validation
    for epoch in range(start_epoch, end_epoch):
        print('epoch: %d/%d' % (epoch, end_epoch-1))
        t0 = time.time()
        # training
        epoch_loss = train_one_epoch(train_dataloader, model, optimizer, device)
        t1 = time.time()
        print('loss=%.4f (took %.2f sec)' % (epoch_loss, t1-t0))
        lr_scheduler.step(epoch_loss)
        # validation
        val_epoch_loss = val_one_epoch(val_dataloader, model, confusion_matrix, device)
        val_epoch_iou = confusion_matrix.get_iou()
        val_epoch_mean_iou = confusion_matrix.get_mean_iou()
        val_epoch_pix_accuracy = confusion_matrix.get_pix_acc()
        
        print('[validation] loss=%.4f, mean iou=%.4f, pixel accuracy=%.4f' % 
              (val_epoch_loss, val_epoch_mean_iou, val_epoch_pix_accuracy))
        print('class IoU: [' + ', '.join([('%.4f' % (x)) for x in val_epoch_iou]) + ']')
        # saving the best status into a weight file
        if val_epoch_pix_accuracy > best_accuracy:
             best_weight_file = Path('weights')/(name + '_best.pth')
             best_accuracy = val_epoch_pix_accuracy
             state = {'model': model.state_dict(), 'epoch': epoch, 'best_accuracy': best_accuracy}
             torch.save(state, best_weight_file)
             print('best accuracy=>saved\n')
        # saving the current status into a weight file
        state = {'model': model.state_dict(), 'epoch': epoch, 'best_accuracy': best_accuracy}
        torch.save(state, weight_file)
        # wandb logging
        wandb.log({'train_loss': epoch_loss, 'val_loss': val_epoch_loss, 'val_accuracy': val_epoch_pix_accuracy})
        
def train_one_epoch(train_dataloader, model, optimizer, device):
    model.train()
    losses = [] 
    for i, (imgs, targets, _) in enumerate(train_dataloader):
        imgs, targets = imgs.to(device), targets.to(device)
        preds = model(imgs)    # forward 
        loss = ce_loss(preds, targets) # calculates the iteration loss  
        optimizer.zero_grad()   # zeros the parameter gradients
        loss.backward()         # backward
        optimizer.step()        # update weights
        print('\t iteration: %d/%d, loss=%.4f' % (i, len(train_dataloader)-1, loss))    
        losses.append(loss.item())
    return torch.tensor(losses).mean().item()


def val_one_epoch(val_dataloader, model, confusion_matrix, device):
    model.eval()
    losses = []
    total = 0
    for i, (imgs, targets, img_file) in enumerate(val_dataloader):
        imgs, targets = imgs.to(device), targets.to(device)
        with torch.no_grad():
            preds = model(imgs)  # forward, preds: (B, 2, H, W)
            loss = ce_loss(preds, targets)
            losses.append(loss.item())
            confusion_matrix.process_batch(preds, targets)
            total += preds.size(0)
            # sample images
            if i == 0:
                preds = torch.argmax(preds, axis=1) # (1, H, W)  
                for j in range(3):
                    save_file = os.path.join('outputs', 'val_%d.png' % (j))
                    plot_image(imgs[j], preds[j], save_file)
                
    avg_loss = torch.tensor(losses).mean().item()
    
    return avg_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=250, help='target epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    parser.add_argument('--name', default='ohhan_road', help='name for the run')

    opt = parser.parse_args()

    train(opt)
