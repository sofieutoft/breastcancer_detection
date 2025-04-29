# %%  Imports

import torch
import torch.nn as nn
import kagglehub
import os
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from torchvision import transforms
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torch import optim, nn
from tqdm import tqdm


# %% Getting the Data Paths

path = kagglehub.dataset_download("aryashah2k/breast-ultrasound-images-dataset")
main_dir = path+'/Dataset_BUSI_with_GT'

# %% Getting Samples Scans for Verification

img_path = path+'/Dataset_BUSI_with_GT/malignant/malignant (93).png'
mask_img_path = path + '/Dataset_BUSI_with_GT/malignant/malignant (93)_mask.png'

transform = transforms.ToTensor()


# %% Loading Raw Data


img_list = []
mask_list = []

categories = ['malignant', 'benign', 'normal']

for category in categories:
    category_dir = os.path.join(path, 'Dataset_BUSI_with_GT', category)

    for filename in os.listdir(category_dir):
        if filename.endswith('.png') and '_mask' not in filename:
            img_path = os.path.join(category_dir, filename)
            mask_path = os.path.join(category_dir, filename.replace('.png', '_mask.png'))

            
            img = Image.open(img_path).convert("RGB")
            img = img.resize((256, 256))
            img_tensor = transform(img)

           
            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert("L")
                mask = mask.resize((256, 256))
                mask_tensor = transform(mask)
                mask_tensor = (mask_tensor > 0).float()
            else:
                mask_tensor = torch.zeros((1, 256, 256))

            img_list.append(img_tensor)
            mask_list.append(mask_tensor)
    
print(len(img_list))
print(len(mask_list))

# %% Data for pytorch

class FullDataset(Dataset):
    def __init__(self, image_list, mask_list, transform=None):
        self.images = image_list
        self.masks = mask_list
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        mask = self.masks[idx]
        if self.transform:
            img, mask = self.transform(img, mask)
        return img, mask
    

# %% The Unet Parts

class DoubleConv(nn.Module):
    def __init__(self,in_channels, out_channels):
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1), 
            nn.Dropout(0.4),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv_op(x)
    
class Downsample(nn.Module):
    def __init__(self,in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)

    def forward(self,x):
        down = self.conv(x)
        p = self.pool(down)
        return down, p
    
class Upsample(nn.Module):
    def __init__(self,in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2,stride=2)
        self.conv = DoubleConv(in_channels,out_channels)

    def forward(self,x1,x2):
        x1 = self.up(x1)
        x = torch.cat([x1,x2],1)
        return self.conv(x)

# %% The Unet Itself

class UNet(nn.Module):
    def __init__(self,in_channels, num_classes):
        super().__init__()
        self.down_convolution_1 = Downsample(in_channels, 64)
        self.down_convolution_2 = Downsample(64, 128)
        self.down_convolution_3 = Downsample(128, 256)
        self.down_convolution_4 = Downsample(256, 512)

        self.bottle_neck = DoubleConv(512, 1024)

        self.up_convolution_1 = Upsample(1024,512)
        self.up_convolution_2 = Upsample(512,256)
        self.up_convolution_3 = Upsample(256,128)
        self.up_convolution_4 = Upsample(128,64)

        self.out = nn.Conv2d(in_channels=64, out_channels= num_classes,kernel_size=1)

    def forward(self,x):
        down_1, p1 = self.down_convolution_1(x)
        down_2, p2 = self.down_convolution_2(p1)
        down_3, p3 = self.down_convolution_3(p2)
        down_4, p4 = self.down_convolution_4(p3)

        b = self.bottle_neck(p4)

        up_1 = self.up_convolution_1(b, down_4)
        up_2 = self.up_convolution_2(up_1, down_3)
        up_3 = self.up_convolution_3(up_2, down_2)
        up_4 = self.up_convolution_4(up_3, down_1)

        out = self.out(up_4)
        return out
    
if __name__ == "__main__":
    double_conv = DoubleConv(256,256)
    print(double_conv)

    input_image = torch.rand((1,3,256,256))
    model = UNet(3,10)
    output = model(input_image)
    print(output.size())


# %% Getting Ready for Training

total_size = len(img_list)
train_size = int(0.8 * total_size)
test_size = total_size - train_size

dataset = FullDataset(img_list, mask_list)

# Split
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

#DataLoaders for batching
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Checking the unique values in the first few masks
num_masks_to_check = min(5, len(mask_list))
for i in range(num_masks_to_check):
    unique_values = torch.unique(mask_list[i])
    print(f"Unique values in mask {i}: {unique_values}")

# Calculating the proportion of non-zero pixels in all masks
total_non_zero_pixels = 0
total_pixels_in_masks = 0

for mask_tensor in mask_list:
    non_zero_pixels = torch.sum(mask_tensor > 0).item()
    total_non_zero_pixels += non_zero_pixels
    total_pixels_in_masks += mask_tensor.numel()

if total_pixels_in_masks > 0:
    proportion_non_zero = total_non_zero_pixels / total_pixels_in_masks
    print(f"\nAverage proportion of background pixels in all masks: {proportion_non_zero:.4f}")
else:
    print("\nNo masks found in mask_list.")

# Visualizing a few masks 
num_masks_to_visualize = min(5, len(mask_list))
for i in range(num_masks_to_visualize):
    plt.figure()
    plt.imshow(mask_list[i].squeeze().cpu().numpy(), cmap='gray')
    plt.title(f"Ground Truth Mask {i}")
    plt.show()


# %% Setting up Training

learning_rate = 1e-4
epochs = 200
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = UNet(in_channels=3, num_classes=1).to(device)
optimizer = optim.AdamW(model.parameters(), lr =learning_rate, weight_decay=3e-3)


pos_weight = torch.tensor([10.0]).to(device)  
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

train_losses = []
test_losses = []


# %% Training 

for epoch in tqdm(range()):
    model.train()
    train_running_loss = 0
    for idx, img_mask in enumerate(tqdm(train_loader)):
        img = img_mask[0].float().to(device)
        mask = img_mask[1].float().to(device)

        y_pred = model(img)
        optimizer.zero_grad()

        loss = criterion(y_pred, mask)
        train_running_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    train_loss = train_running_loss/(idx+1)
    train_losses.append(train_loss)

    model.eval()
    test_running_loss = 0
    with torch.no_grad():
        for idx, img_mask in enumerate(tqdm(test_loader)):
            img = img_mask[0].float().to(device)
            mask = img_mask[1].float().to(device)

            y_pred = model(img)
            loss = criterion(y_pred, mask)
            test_running_loss += loss.item()

        test_loss= test_running_loss/(idx+1)
        test_losses.append(test_loss)
    
    print("-"*30)
    print(f"Train Loss EPOCH {epoch+1}: {train_loss:.4f}")
    print(f"Test Loss EPOCH {epoch+1}: {test_loss:.4f}")
    print("-"*30)


# %% SAVE MODEL

save_path = 'C:\\Users\\ianbo\\OneDrive\\Desktop\\Academics\\Spring 2025\\Data Science Principles\\fullUNet#1.pth'
torch.save(model.state_dict(), save_path)
print(f"Trained model saved to {save_path}")


# %% Getting some predicted masks for visualisation

# %% Building the images

def image_rebuilder():
    model.eval()
    
    batch = next(iter(test_loader))
    images, masks = batch

    i = 2
    img = images[i].unsqueeze(0).to(device)  
    true_mask = masks[i].squeeze().cpu()

    
    pred_mask = model(img)  
    print(pred_mask.min().item(), pred_mask.max().item())
    
    pred_mask = pred_mask.squeeze().cpu()  

    
    pred_mask_detached = pred_mask.detach().cpu().numpy()

  
    plt.imshow(pred_mask_detached.squeeze(), cmap='gray')  
    plt.title("Predicted Mask Before Thresholding")
    plt.colorbar()
    plt.show()


    
    pred_mask = (pred_mask > 0.5).float()

    
    img_disp = images[i].permute(1, 2, 0).cpu() 

    
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(img_disp)
    axs[0].set_title('Input Image')
    axs[1].imshow(true_mask, cmap='gray')
    axs[1].set_title('True Mask')
    axs[2].imshow(pred_mask, cmap='gray')
    axs[2].set_title('Predicted Mask')
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.show()


image_rebuilder()

# %% Plotting the Loss Curves
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, epochs + 1), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and Test Loss Curves')
plt.legend()
plt.grid(True)

print("Done")

# %% IoU Score


def calculate_iou(pred_mask, true_mask, threshold=0.5):
   
    model.eval()
    pred_mask_bin = (pred_mask > threshold).astype(np.uint8)
    true_mask_bin = (true_mask > 0.5).astype(np.uint8)

    intersection = np.logical_and(pred_mask_bin, true_mask_bin).sum()
    union = np.logical_or(pred_mask_bin, true_mask_bin).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    iou = intersection / union
    return iou

def get_iou(data_loader):
    iou_scores = []

    with torch.no_grad():
        for batch in data_loader:
            image, mask = batch
            image = image.float().to(device)
            mask = mask.float().to(device)

            y_pred = model(image)
            y_pred = torch.sigmoid(y_pred) 
            y_pred = y_pred.squeeze(1)  

            for i in range(image.size(0)):
                pred_mask = y_pred[i].cpu().numpy()
                true_mask = mask[i].cpu().numpy()
                iou = calculate_iou(pred_mask, true_mask, threshold=0.5)
                iou_scores.append(iou)
    return np.mean(iou_scores)

print(f'Average IoU Score for training set: {np.mean(get_iou(train_loader)):.4f}')
print(f'Average IoU Score for test set: {np.mean(get_iou(test_loader)):.4f}')
