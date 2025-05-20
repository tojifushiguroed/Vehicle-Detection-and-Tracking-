import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CocoDetectionTransform(CocoDetection):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        img = F.to_tensor(img)
        new_target = []
        for obj in target:
            if 'bbox' in obj:
                bbox = obj['bbox']
                xmin, ymin, w, h = bbox
                xmax = xmin + w
                ymax = ymin + h
                new_target.append({
                    'boxes': torch.tensor([xmin, ymin, xmax, ymax], dtype=torch.float32),
                    'labels': torch.tensor(obj['category_id'], dtype=torch.int64)
                })
        boxes = torch.stack([t['boxes'] for t in new_target])
        labels = torch.stack([t['labels'] for t in new_target])
        target_out = {'boxes': boxes, 'labels': labels}
        if self.transforms:
            img = self.transforms(img)
        elif not isinstance(img,torch.Tensor):
            img = F.to_tensor(img)  

        return img, target_out


train_dir = r"C:\Users\DELL\Downloads\UA-DETRAC-DATASET-10K.v2-2024-11-14-3-48pm.coco\train\images"
train_ann = r"C:\Users\DELL\Downloads\UA-DETRAC-DATASET-10K.v2-2024-11-14-3-48pm.coco\train\_annotations.coco.json"
val_dir = r"C:\Users\DELL\Downloads\UA-DETRAC-DATASET-10K.v2-2024-11-14-3-48pm.coco\valid\images"
val_ann = r"C:\Users\DELL\Downloads\UA-DETRAC-DATASET-10K.v2-2024-11-14-3-48pm.coco\valid\_annotations.coco.json"

train_data = CocoDetectionTransform(train_dir, train_ann)
val_data = CocoDetectionTransform(val_dir, val_ann)

train_loader = DataLoader(train_data, batch_size=8, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_data, batch_size=8, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

model = fasterrcnn_resnet50_fpn(pretrained=False)  
checkpoint = torch.load(r"C:\Users\DELL\source\repos\VisionProject\VisionProject\fasterrcnn_vehicle_detector.pth")
model.load_state_dict(checkpoint)
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

scaler = GradScaler()

def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0
    for images, targets in tqdm(loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        with autocast():
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += losses.item()
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader):
    model.eval() 
    val_loss = 0
    for images, targets in tqdm(loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        model.train()  
        loss_dict = model(images, targets)
        model.eval()   

        losses = sum(loss for loss in loss_dict.values())
        val_loss += losses.item()
    return val_loss / len(loader)

train_losses, val_losses = [], []
num_epochs = 50
log_file = open("training_log.txt", "w")
log_file.write("Epoch,Train Loss,Validation Loss\n")
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    train_loss = train_one_epoch(model, train_loader, optimizer)
    val_loss = evaluate(model, val_loader)
    scheduler.step()

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    log_line = f"{epoch+1},{train_loss:.4f},{val_loss:.4f}\n"
    log_file.write(log_line)
    log_file.flush() 
log_file.close()
torch.save(model.state_dict(), 'fasterrcnn_vehicle_detector.pth')

plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.legend()
plt.title('Training vs Validation Loss')
plt.savefig("loss_curve.png")