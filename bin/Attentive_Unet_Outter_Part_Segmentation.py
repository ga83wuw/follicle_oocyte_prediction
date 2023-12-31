

import os
import copy
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch import nn
import cv2
from tensorflow.keras.preprocessing.image import smart_resize
import torchvision.transforms as transforms
import random
from itertools import filterfalse
import torch
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.model_selection import train_test_split
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms import InterpolationMode
import numpy as np
from scipy.spatial import ConvexHull
from scipy.ndimage import binary_fill_holes
import skimage
import torch.nn as nn
import torch.nn.functional as F

#from google.colab import drive (if using google collaboratory)
#drive.mount('/content/drive', force_remount=True)

# We define here the paths to access the data, first PATH is for the pretraining data, PATH2 is for the fine-tuning data
# You can change the paths here to adapt to your image location

PATH = 'drive/MyDrive/IIIA-CSIC/data/images/external_dataset/'
IMAGE_PATH = os.path.join(PATH, 'input')
MASK_PATH = os.path.join(PATH, 'mask')

PATH2 = '/content/drive/MyDrive/IIIA-CSIC/data/images/outer_finetuning/'
TUNE_IMAGE_PATH = os.path.join(PATH2, 'roi_images')
TUNE_MASK_PATH = os.path.join(PATH2, 'roi_masks')

# Hyperparameters definition
IMAGE_SIZE=192
SHUFFLE_DATA = True

# Define here were you want to save the weights
PRETRAINING_WEIGHTS_PATH='/content/drive/MyDrive/IIIA-CSIC/weights/outter_pretraining_weights_unet.pth'
FINETUNING_WEIGHTS_PATH='/content/drive/MyDrive/IIIA-CSIC/weights/outter_shape_aware_finetuning_weights_unet.pth'

"""# Data Processing et Générateur"""

def contrast(im,val):
  '''
  increase the contrast of an image by the given value 'val'
  '''

  im=(im-torch.mean(im))*val+torch.mean(im)
  return(im)

def replace_black_pixels(image, color):
    '''
    Raplace the black pixels of an image by the color given as argument
    '''

    black_pixels = (image == 0)  # Pixels noirs
    image[black_pixels] = color  # Remplacer les pixels noirs par la couleur

    return image

#We define the data generator for the neural network training

class DataGenerator(Dataset):
    def __init__(self, image_folder, mask_folder, names,mask_inversion,pretrain,data_augm):
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.names = names
        self.mask_inversion=mask_inversion
        self.pretrain=pretrain
        self.data_augm=data_augm
        self.image_size=IMAGE_SIZE

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]

        image_path = os.path.join(self.image_folder, name)
        mask_path = os.path.join(self.mask_folder, name)

        # resizing + normalization
        image = torch.tensor(smart_resize(cv2.imread(image_path),(self.image_size,self.image_size))[:,:,0])/255
        mask = torch.tensor(smart_resize(cv2.imread(mask_path),(self.image_size,self.image_size))[:,:,0])/255


        # data_augmentation if data_augm=True
        if self.data_augm:
          val = random.uniform(1, 3)
          image=contrast(image,val) #contrast data augm
          im_bis=copy.copy(image)
          random_number = random.randint(0, 360) #rotation data augm
          transform = T.RandomRotation((random_number,random_number),interpolation=InterpolationMode.BILINEAR,fill=image[0,0].item())
          image = transform(image.unsqueeze(0)).squeeze(0)
          image = replace_black_pixels(image, im_bis[0,0])
          mask =  transform(mask.unsqueeze(0)).squeeze(0)

        if self.mask_inversion:
            mask = torch.tensor(np.where(mask >= 0.5, 0, 1))
        else:
            mask = torch.tensor(np.where(mask >= 0.5, 1, 0))

        e=mask.numpy()
        if self.pretrain:
          a=create_filled_convex_hull(e)[:,:,0] #filling of the pretraining mask them like a filled disk
          mask=torch.tensor(create_filled_convex_hull(e)[:,:,0])


        image=torch.clamp(image,0,1)
        image = image.clone().detach()
        mask = mask.clone().detach()

        X = image.unsqueeze(0)
        Y = mask.unsqueeze(0)

        return X, Y

def create_generators(image_folder, mask_folder,mask_inversion=False,pretrain=True,train_ratio = 0.7,val_ratio = 0.2,test_ratio = 0.1,data_augm=True):
    image_names = os.listdir(image_folder)
    mask_names = os.listdir(mask_folder)
    common_names = list(set(image_names) & set(mask_names))

    total_names = common_names
    total_count = len(total_names)

    slice1= round(total_count * train_ratio)
    slice2 = round(total_count * (val_ratio+train_ratio))

    if pretrain:
      BATCH_SIZE = 16  #batch_size for pretraining
    else:
      BATCH_SIZE = 4   #reduced batch size for fine-tuning (because only a few images)


    train_names = total_names[:slice1]
    val_names = total_names[slice1:slice2]
    test_names = total_names[slice2:]
    train_generator = DataGenerator(image_folder, mask_folder, train_names,mask_inversion,pretrain,data_augm)
    val_generator = DataGenerator(image_folder, mask_folder, val_names,mask_inversion,pretrain,data_augm)
    test_generator = DataGenerator(image_folder, mask_folder, test_names,mask_inversion,pretrain,data_augm)

    train_loader = DataLoader(train_generator, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_generator, batch_size=VAL_BATCH_SIZE, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_generator, batch_size=TEST_BATCH_SIZE, shuffle=False, drop_last=True)

    return train_loader, val_loader, test_loader

# Initialization of the generators
train_loader, val_loader, test_loader = create_generators(IMAGE_PATH,MASK_PATH, train_ratio=0.7, val_ratio=0.3, test_ratio=0.0)

"""# Show batch"""

def create_filled_convex_hull(mask):
    '''
    makes the mask convex and fill holes
    '''
    indices = np.argwhere(mask == 1)
    hull = ConvexHull(indices)
    image = np.zeros((IMAGE_SIZE,IMAGE_SIZE, 1), dtype=np.uint8)
    contour = indices[hull.vertices]

    contour[:, [0, 1]] = contour[:, [1, 0]]

    color = (1, 1, 1)
    cv2.fillPoly(image, [contour], color)

    return image

def local_entropy(im, kernel_size=3, normalize=True, num_points=30):
    kernel = skimage.morphology.disk(kernel_size)
    print(np.min(im),np.max(im))
    entr_img = skimage.filters.rank.entropy(skimage.util.img_as_ubyte(im), kernel)

    if normalize:
        max_img = np.max(entr_img)
        entr_img = (entr_img * 255 / max_img).astype(np.uint8)

    _, mask1 = cv2.threshold(entr_img, 120, 255, cv2.THRESH_BINARY)
    _, mask2 = cv2.threshold(entr_img, 190, 255, cv2.THRESH_BINARY)


    # Find contours in the mask
    contours, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Sample a subset of points from the contour
    contour_points = largest_contour.squeeze()
    sampled_points = contour_points[np.linspace(0, len(contour_points)-1, num=num_points, dtype=int)]

    # Create a new contour from the sampled points
    sampled_contour = sampled_points.reshape((-1, 1, 2))

    # Calculate the size of the largest contour
    contour_size = cv2.contourArea(sampled_contour)

    # Calculate the final result by dividing the total surface by the length of the contour
    final_result = np.sum(mask2[mask2 == 255])/contour_size

    # Display the image and contour
    im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im_with_contour = cv2.drawContours(im_rgb, [sampled_contour], -1, (255, 0, 0), 2)

    return mask2, im_with_contour, final_result

import numpy as np

def dark_pixels(image, seuil=0.4):
    nouvelle_image = np.where(image > seuil*np.max(image),0, 255).astype(np.uint8)
    kernel = np.ones((5,5),np.uint8)
    new=cv2.erode(new,kernel,iterations=2)
    return nouvelle_image

# plot some maks and images

train_loader,val_generator,test_generator = create_generators(TUNE_IMAGE_PATH, TUNE_MASK_PATH,mask_inversion=True,pretrain=False, train_ratio=0.8, val_ratio=0.2, test_ratio=0.0)
itera = iter(train_loader)
batch = next(itera)

# Extract images (X) and segmentation maks (Y) from the batch
X, Y = batch

for i in range(len(X)):
    image = (X[i]).squeeze(0).numpy()
    mask = (Y[i]).squeeze(0).numpy()
    mask_copy = copy.copy(mask)

    plt.figure(figsize=(10, 8))

    # plot mask
    plt.subplot(2, 2, 3)
    plt.imshow(mask)
    plt.axis('off')

    # plot image
    plt.subplot(2, 2, 4)
    plt.imshow(image)
    plt.axis('off')

    plt.show()

"""# MODEL U-net"""

# We define there our Unet model structure


class Down(nn.Module):
    # Contracting Layer

    def __init__(self, in_channels, out_channels, dropout_probability):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding='same'),
            nn.Dropout(dropout_probability),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding='same'),
            nn.Dropout(dropout_probability),
            nn.BatchNorm2d(out_channels), #2D ou 3D ici ?
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    # Expanding Layer

    def __init__(self, in_channels, out_channels, dropout_probability):
        super().__init__()

        self.up = nn.Upsample(scale_factor=(2, 2), mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding='same'),
            nn.Dropout(dropout_probability),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding='same'),
            nn.Dropout(dropout_probability),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)

import torch.nn.functional as F
class Unet(nn.Module):

    def __init__(self, input_channels, output_classes, hidden_channels, dropout_probability,kernel_size):
        super(Unet, self).__init__()

        # Initial Convolution Layer
        self.inc = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=kernel_size, padding='same'),
            nn.Dropout(dropout_probability),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding='same'),
            nn.Dropout(dropout_probability),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True))

        # Contracting Path
        self.down1 = Down(hidden_channels, 2 * hidden_channels, dropout_probability)
        self.down2 = Down(2 * hidden_channels, 4 * hidden_channels, dropout_probability)
        self.down3 = Down(4 * hidden_channels, 4 * hidden_channels, dropout_probability) #modif hidden_channels ici !!!
        #self.down4 = Down(8 * hidden_channels, 8 * hidden_channels, dropout_probability)

        # Expanding Path
        #self.up1 = Up(16 * hidden_channels, 4 * hidden_channels, dropout_probability)
        self.up2 = Up(8 * hidden_channels, 2 * hidden_channels, dropout_probability)
        self.up3 = Up(4 * hidden_channels, hidden_channels, dropout_probability)
        self.up4 = Up(2 * hidden_channels, hidden_channels, dropout_probability)

        # Output Convolution Layer
        self.outc = nn.Conv2d(hidden_channels, output_classes, kernel_size=1) #3D ou 2D ici ??
        self.softmax = nn.Sigmoid() #sigmoid pas softmax ici
        #self.softmax = nn.ReLU()

    def forward(self, x):
        # Initial Convolution Layer
        x1 = self.inc(x)

        # Contracting Path
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x7 = self.up2(x4, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)

        # Output Convolution Layer
        logits = self.outc(x9)
        output = self.softmax(logits)
        return output

#We define there the Unet, but this time with an attention mechanism

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super(ConvBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super(UpConv, self).__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, n_coefficients, hidden_channels):
        super(AttentionBlock, self).__init__()

        self.W_gate = nn.Sequential(
            nn.Conv2d(F_g, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(n_coefficients, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip_connection):
        g1 = self.W_gate(gate)
        x1 = self.W_x(skip_connection)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = skip_connection * psi
        return out


class AttentionUNet(nn.Module):
    def __init__(self, img_ch=1, output_ch=1, hidden_channels=16):
        super(AttentionUNet, self).__init__()

        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = ConvBlock(img_ch, hidden_channels, hidden_channels*2)
        self.Conv2 = ConvBlock(hidden_channels, hidden_channels*2, hidden_channels*4)
        self.Conv3 = ConvBlock(hidden_channels*2, hidden_channels*4, hidden_channels*8)
        self.Conv4 = ConvBlock(hidden_channels*4, hidden_channels*8, hidden_channels*16)
        self.Conv5 = ConvBlock(hidden_channels*8, hidden_channels*16, hidden_channels*32)

        self.Up5 = UpConv(hidden_channels*16, hidden_channels*8, hidden_channels*32)
        self.Att5 = AttentionBlock(F_g=hidden_channels*8, F_l=hidden_channels*8, n_coefficients=hidden_channels*16, hidden_channels=hidden_channels)
        self.UpConv5 = ConvBlock(hidden_channels*16, hidden_channels*8, hidden_channels*16)

        self.Up4 = UpConv(hidden_channels*8, hidden_channels*4, hidden_channels*16)
        self.Att4 = AttentionBlock(F_g=hidden_channels*4, F_l=hidden_channels*4, n_coefficients=hidden_channels*8, hidden_channels=hidden_channels)
        self.UpConv4 = ConvBlock(hidden_channels*8, hidden_channels*4, hidden_channels*8)

        self.Up3 = UpConv(hidden_channels*4, hidden_channels*2, hidden_channels*8)
        self.Att3 = AttentionBlock(F_g=hidden_channels*2, F_l=hidden_channels*2, n_coefficients=hidden_channels*4, hidden_channels=hidden_channels)
        self.UpConv3 = ConvBlock(hidden_channels*4, hidden_channels*2, hidden_channels*4)

        self.Up2 = UpConv(hidden_channels*2, hidden_channels, hidden_channels*4)
        self.Att2 = AttentionBlock(F_g=hidden_channels, F_l=hidden_channels, n_coefficients=hidden_channels*2, hidden_channels=hidden_channels)
        self.UpConv2 = ConvBlock(hidden_channels*2, hidden_channels, hidden_channels*2)

        self.Conv = nn.Conv2d(hidden_channels, output_ch, kernel_size=1, stride=1, padding=0)
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.MaxPool(e1)
        e2 = self.Conv2(e2)

        e3 = self.MaxPool(e2)
        e3 = self.Conv3(e3)

        e4 = self.MaxPool(e3)
        e4 = self.Conv4(e4)

        e5 = self.MaxPool(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        s4 = self.Att5(gate=d5, skip_connection=e4)
        d5 = torch.cat((s4, d5), dim=1)
        d5 = self.UpConv5(d5)

        d4 = self.Up4(d5)
        s3 = self.Att4(gate=d4, skip_connection=e3)
        d4 = torch.cat((s3, d4), dim=1)
        d4 = self.UpConv4(d4)

        d3 = self.Up3(d4)
        s2 = self.Att3(gate=d3, skip_connection=e2)
        d3 = torch.cat((s2, d3), dim=1)
        d3 = self.UpConv3(d3)

        d2 = self.Up2(d3)
        s1 = self.Att2(gate=d2, skip_connection=e1)
        d2 = torch.cat((s1, d2), dim=1)
        d2 = self.UpConv2(d2)

        out = self.Conv(d2)
        out = self.sigm(out)

        return out

"""# TRAIN"""

#we define here the classical segmentation Dice Loss
class DiceLoss(nn.Module):
    def __init__(self, weight = None, size_average = True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth = 1.):


        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice

# Loss to encourage elimination of mask artifacts
class CustomLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CustomLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1., penalty_weight=10.):
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        # Convert to NumPy
        inputs_np = inputs.cpu().detach().numpy().astype(bool)
        targets_np = targets.cpu().detach().numpy().astype(bool)

        # Appliquer la fonction de connectivité pour détecter les composantes connexes
        labeled_inputs, num_components = ndimage.label(inputs_np)

        # Calculer la pénalité si le masque a plusieurs composantes connexes
        penalty = penalty_weight * float(num_components)*float(num_components)

        # Calculer la loss en multipliant la Dice loss par le terme de pénalité
        loss = (1 - dice) * torch.tensor(penalty, device=inputs.device)

        return loss

class CustomLoss2(nn.Module):
    '''
    loss to penalize non-convex contours
    '''
    def __init__(self, weight=None, size_average=True):
        super(CustomLoss2, self).__init__()

    def forward(self, inputs, targets, smooth=1., penalty_weight=10.):
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        # Convert tensors to numpy arrays
        inputs_np = inputs.cpu().detach().numpy().astype(float)
        targets_np = targets.cpu().detach().numpy().astype(float)
        inputs_np= np.where(inputs_np >= 0.5, 1, 0)

        # Calculate the convexity score
        convexity_score = self.calculate_convexity_score(inputs_np)
        convexity_score = convexity_score
        #print("convex: ",convexity_score)

        # Calculate the loss by multiplying the Dice loss by the penalty term
        loss = (1 - dice) * torch.tensor(convexity_score, device=inputs.device)
        loss=torch.sum(loss)/ inputs.size(0)

        return loss

    def calculate_convexity_score(self, mask):
        scores = []
        for m in mask:
            m=m[0]
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #m.astype(np.uint8)

            if len(contours) == 0:
                scores.append(0.0)
                continue

            flattened_points = np.concatenate(contours, axis=0)
            contour = flattened_points.reshape((-1, 2))
            contour_area = cv2.contourArea(contour)

            if contour_area == 0:
                scores.append(0.0)
                continue

            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            convexity_score = 1.0 - abs(contour_area - hull_area) / contour_area
            scores.append(convexity_score)

        return np.array(scores)

"""
#code for trying the loss on simple cases
# Create target mask with a single circular mask
target_mask = np.zeros((256, 256), dtype=np.float32)
radius = 80
center_x, center_y = 128, 128
y_indices, x_indices = np.ogrid[:target_mask.shape[0], :target_mask.shape[1]]
mask = (x_indices - center_x) ** 2 + (y_indices - center_y) ** 2 <= radius ** 2
target_mask[mask] = 1.0

# Create predicted mask with two circular masks intersecting
predicted_mask = np.zeros((256, 256), dtype=np.float32)
radius = 60
center1_x, center1_y = 104, 104
center2_x, center2_y = 152, 152
mask1 = (x_indices - center1_x) ** 2 + (y_indices - center1_y) ** 2 <= radius ** 2
mask2 = (x_indices - center2_x) ** 2 + (y_indices - center2_y) ** 2 <= radius ** 2
predicted_mask[mask1] = 1.0
predicted_mask[mask2] = 1.0

# Convert numpy arrays to tensors
target_tensor = torch.from_numpy(target_mask).float()
predicted_tensor = torch.from_numpy(predicted_mask).float()


# Instantiate the custom loss function
loss_function = CustomLoss2()

# Calculate the loss for the two masks
loss_target = loss_function(predicted_tensor, target_tensor)

# Create predicted mask with two circular masks non-intersecting
predicted_mask_non_intersecting = np.zeros((256, 256), dtype=np.float32)
radius = 60
center1_x, center1_y = 80, 80
center2_x, center2_y = 150, 150
mask1 = (x_indices - center1_x) ** 2 + (y_indices - center1_y) ** 2 <= radius ** 2
mask2 = (x_indices - center2_x) ** 2 + (y_indices - center2_y) ** 2 <= radius ** 2
predicted_mask_non_intersecting[mask1] = 1.0
predicted_mask_non_intersecting[mask2] = 1.0

# Convert numpy array to tensor
predicted_tensor_non_intersecting = torch.from_numpy(predicted_mask_non_intersecting).float()

# Calculate the loss for the non-intersecting masks
loss_non_intersecting = loss_function(predicted_tensor_non_intersecting, target_tensor)

# Print the losses
print("Loss (Intersecting Masks):", loss_target.item())
print("Loss (Non-Intersecting Masks):", loss_non_intersecting.item())

# Display the masks
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(target_mask, cmap='gray')
axes[0].set_title('Target Mask')
axes[0].axis('off')
axes[1].imshow(predicted_mask, cmap='gray')
axes[1].set_title('Predicted Mask (Intersecting)')
axes[1].axis('off')
axes[2].imshow(predicted_mask_non_intersecting, cmap='gray')
axes[2].set_title('Predicted Mask (Non-Intersecting)')
axes[2].axis('off')
plt.show()
"""

def resample_contour(contour, num_points):
    '''
    makes the contour contains the number of points given
    '''
    contour_length = contour.shape[0]
    indices = np.linspace(0, contour_length - 1, num_points, dtype=int)
    resampled_contour = contour[indices]
    return resampled_contour

class ShapeAwareLoss(nn.Module):
    '''
    Loss which is the product of dice loss and euclidean distance between predicted and target contours.
    Encourage model to focus on contours.
    '''
    def __init__(self, num_points=100):
        super(ShapeAwareLoss, self).__init__()
        self.num_points = num_points

    def forward(self, pred_mask, target_mask):
        # Compute binary cross entropy loss

        loss1 = DiceLoss()
        bce_loss=loss1(pred_mask,target_mask)
        #bce_loss = F.binary_cross_entropy_with_logits(pred_mask, target_mask, reduction='mean')

        # Convert predicted mask and target mask to numpy arrays
        pred_mask_np = pred_mask.detach().cpu().numpy()
        target_mask_np = target_mask.detach().cpu().numpy()

        # Apply sigmoid activation and convert to binary images
        pred_mask_binary = pred_mask > 0.5 #torch .sigmoid before
        target_mask_binary = target_mask > 0.5

        # Iterate over the batch dimension
        batch_loss = 0.0
        for pred_mask_batch, target_mask_batch in zip(pred_mask_binary, target_mask_binary):
            # Convert binary images to numpy arrays

            pred_mask_np_batch = pred_mask_batch.squeeze().cpu().numpy().astype(np.uint8)
            target_mask_np_batch = target_mask_batch.squeeze().cpu().numpy().astype(np.uint8)



            # Find contours in the predicted mask
            pred_contours, _ = cv2.findContours(pred_mask_np_batch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Find contours in the target mask
            target_contours, _ = cv2.findContours(target_mask_np_batch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


            if target_contours!=[] or pred_contours!=[]:
                # Select the largest contour from each list
                try:
                  pred_contour = max(pred_contours, key=cv2.contourArea)
                  target_contour = max(target_contours, key=cv2.contourArea)

                  # Resample the contours to have the same number of points
                  pred_contour_resampled = resample_contour(pred_contour, self.num_points)
                  target_contour_resampled = resample_contour(target_contour, self.num_points)

                  # Convert contours to NumPy arrays
                  pred_contour_np = np.array(pred_contour_resampled)
                  target_contour_np = np.array(target_contour_resampled)
                  #contour_length = cv2.arcLength(target_contour_resampled, closed=True)
                  #if contour_length==0:
                  #contour_length=400
                  #print("con leght:",contour_length)

                  # Calculate the difference in surface area between the mask and its convex hull
                  #pred_hull_area = cv2.contourArea(cv2.convexHull(pred_contour_np))
                  #print("pred",pred_hull_area)
                  #target_hull_np=cv2.convexHull(pred_contour_np)
                  #target_hull_area = cv2.contourArea(pred_contour_np)
                  #print("target",target_hull_area)

                  #area_diff = abs(pred_hull_area - target_hull_area)/pred_hull_area

                  #plt.figure()
                  #plt.plot(pred_contour_np[:, 0, 0], pred_contour_np[:, 0, 1], 'r', label='Predicted Contour')
                  #plt.plot(target_contour_np[:, 0, 0], target_contour_np[:, 0, 1], 'y', label='Target Contour')
                  #plt.legend()
                  #plt.show()

                  # Compute Euclidean distance between the predicted and target contours
                  distance = np.linalg.norm(pred_contour_np - target_contour_np)
                except:
                  distance=1
                  area_diff=0.1

            else:
              distance=1
              area_diff=0.1

            # Accumulate the batch loss
            batch_loss += bce_loss*distance

        # Compute the average loss over the batch
        shape_aware_loss = batch_loss / pred_mask.size(0)

        return shape_aware_loss

#Pretraining Phase

NB_EPOCHS=100
seed = 123
torch.manual_seed(seed)

model = AttentionUNet() #UNet(input_channels=1, output_classes=1, hidden_channels=32, dropout_probability=0,kernel_size=(3,3))
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")
train_generator,val_generator,test_generator = create_generators(IMAGE_PATH, MASK_PATH,train_ratio=0.8,val_ratio=0.2,test_ratio=0.0,mask_inversion=False,pretrain=True)

def train():
    # Use gpu or cpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print('device:', device)

    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print("Nombre total de paramètres : ", total_params)

    #choice of the loss
    criterion = DiceLoss()
    #criterion = ShapeAwareLoss(num_points=30)
    #criterion2 = ContourLoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    ###############################################################
    # Start Training                                              #
    ###############################################################
    model.train()

    train_losses = []
    val_losses = []

    for epoch in range(1, NB_EPOCHS + 1):
        print('epoch:', epoch)
        train_loss = []
        model.train()
        train_range = tqdm(train_generator)

        for (X, Y) in train_range:
            X = X.to(torch.float).to(device)
            Y = Y.to(torch.float).to(device)
            optimizer.zero_grad()
            S2_pred = model(X)
            loss = criterion(S2_pred,Y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            train_range.set_description("TRAIN -> epoch: %4d || loss: %4.4f" % (epoch, np.mean(train_loss)))
            train_range.refresh()

        train_losses.append(np.mean(train_loss))

        ###############################################################
        # Start Evaluation                                            #
        ###############################################################

        model.eval()
        val_loss = []
        with torch.no_grad():
            for (image, target) in tqdm(val_generator, desc='validation'):
                image = image.to(device).to(torch.float)
                y_true = target.to(device).to(torch.float)
                y_true= y_true.to(torch.float).to(device)
                y_pred = model(image)
                loss = criterion(y_pred,y_true)
                val_loss.append(loss.item())

        val_losses.append(np.mean(val_loss))
        print("Val_Loss:", np.mean(val_loss))

    plt.figure()
    epochs = range(1, NB_EPOCHS + 1)
    plt.plot(epochs, train_losses, label='train')
    plt.plot(epochs, val_losses, label='val')

#launch the pretraining phase
train()

#save the pretraining weights in a file
torch.save(model.state_dict(),PRETRAINING_WEIGHTS_PATH)

#print one result

itera=iter(val_generator)
image, mask = next(itera)
image = image.to('cuda')
model.to('cuda')
prediction = model(image).to('cuda')
pred2=copy.copy(prediction)
prediction = prediction.squeeze().cpu().detach().numpy()[0]

# Appliquer le seuil (threshold)
prediction = np.where(prediction >= 0.5, 1, 0)

# Afficher l'image d'entrée, le masque réel et la prédiction
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(image.squeeze().cpu().detach().numpy()[0], cmap="gray")
plt.title("Image d'entrée")
plt.subplot(1, 3, 2)
plt.imshow(mask.squeeze().cpu().detach().numpy()[0], cmap="gray")
plt.title("Masque réel")
plt.subplot(1, 3, 3)
plt.imshow(prediction, cmap="gray")
plt.title("Prédiction")
plt.show()

"""**Fine-Tuning Phase **"""

# Load the saved weights of the pretraining
model = AttentionUNet()
model.load_state_dict(torch.load(PRETRAINING_WEIGHTS_PATH))

# We use more epochs for the finetuning phase
# Pretain argument = False
# We can change the ratio of of training/validation/test params with these parameters: train_ratio=1.0, val_ratio=0.0, test_ratio=0.0


NB_EPOCHS=300
torch.manual_seed(112)

train_generator,val_generator,test_generator = create_generators(TUNE_IMAGE_PATH, TUNE_MASK_PATH,mask_inversion=True,pretrain=False, train_ratio=1.0, val_ratio=0.0, test_ratio=0.0)

def train():
    # Use gpu or cpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print('device:', device)

    model.to(device)

    # Loss definition
    #criterion=DiceLoss()
    #criterion=CustomLoss2()
    criterion = ShapeAwareLoss(num_points=100)


    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    ###############################################################
    # Start Training                                              #
    ###############################################################
    model.train()

    train_losses = []
    val_losses = []

    for epoch in range(1, NB_EPOCHS + 1):
        print('epoch:', epoch)
        train_loss = []
        model.train()
        train_range = tqdm(train_generator)

        for (X, Y) in train_range:
            X = X.to(torch.float).to(device)
            Y = Y.to(torch.float).to(device)
            optimizer.zero_grad()
            S2_pred = model(X)
            loss = criterion(S2_pred,Y)
            loss.backward()

            optimizer.step()
            train_loss.append(loss.item())
            train_range.set_description("TRAIN -> epoch: %4d || loss: %4.4f" % (epoch, np.mean(train_loss)))
            train_range.refresh()

        train_losses.append(np.mean(train_loss))

        ###############################################################
        # Start Evaluation                                            #
        ###############################################################

        model.eval()
        val_loss = []
        with torch.no_grad():
            for (image, target) in tqdm(val_generator, desc='validation'):
                image = image.to(device).to(torch.float)
                y_true = target.to(device).to(torch.float)
                y_pred = model(image)
                loss = criterion(y_pred,y_true)
                val_loss.append(loss.item())

        val_losses.append(np.mean(val_loss))
        print("Val_Loss:", np.mean(val_loss))

    plt.figure()
    epochs = range(1, NB_EPOCHS + 1)
    plt.plot(epochs, train_losses, label='train')
    plt.plot(epochs, val_losses, label='val')

#fine-tuning the network
train()

#save the weights of the fine-tuned Unet
torch.save(model.state_dict(),FINETUNING_WEIGHTS_PATH)

#load the saved fine-tuning weights
model.load_state_dict(torch.load(FINETUNING_WEIGHTS_PATH)

# show results of the inference on one batch
itera=iter(train_generator) #pass the generator as argument there
image, mask = next(itera)

model.to('cuda')
#image=contrast(image,4)
image = image.to('cuda')
prediction = model(image).to('cuda')
prediction = prediction.squeeze().cpu().detach().numpy()

# Apply threshold
prediction = np.where(prediction >= 0.5, 1, 0)

# Calculate Dice Loss for all the batch
dice_loss = DiceLoss()
dice_scores = []
percent_diff_scores = []
for i in range(image.shape[0]):
    dice = dice_loss(torch.from_numpy(prediction[i]), mask[i])
    dice_scores.append(dice.item())

    # Calculate the percentage of difference between predicted and target area
    pred_surface = np.sum(prediction[i])
    true_surface = torch.sum(mask[i]).item()
    percent_diff = (pred_surface - true_surface) / true_surface * 100
    percent_diff_scores.append(percent_diff)

# Plot images, target masks and predicted mask
batch_size = image.shape[0]
fig, axs = plt.subplots(batch_size, 4, figsize=(12, 4*batch_size))
for i in range(batch_size):
    im5=image[i].squeeze().cpu().detach().numpy()
    k=local_entropy(im5)
    print(k[2])
    axs[i, 0].imshow(image[i].squeeze().cpu().detach().numpy(), cmap="gray")
    axs[i, 0].set_title("Image d'entrée")
    axs[i, 1].imshow(mask[i].squeeze().cpu().detach().numpy(), cmap="gray")
    axs[i, 1].set_title("Masque réel")
    axs[i, 2].imshow(prediction[i], cmap="gray")
    axs[i, 2].set_title("Prédiction")
    #axs[i, 3].imshow(k[0])
    #axs[i, 3].set_title("Entropy")


plt.tight_layout()
plt.show()

# Calculate mean of the Dice Loss and mean of percentage difference on the batch
mean_dice_score = np.mean(dice_scores)
mean_percent_diff = np.mean(percent_diff_scores)
std_percent_diff=np.std(percent_diff_scores)

print("Dice Loss (mean on the batch) :", mean_dice_score)
print("Percentage of difference of surface (mean on the batch) :", mean_percent_diff)
print("Standard deviation of percentage diff",std_percent_diff)

"""Convex Correction"""

itera=iter(val_generator)

#results but with convex correction of the masks as post-processing to see if it improves the results and metrics

image, mask = next(itera)

# Passer l'image à travers le modèle pour obtenir la prédiction
image = image.to('cuda')
prediction = model(image)
prediction = prediction.squeeze().cpu().detach().numpy()

# Appliquer le seuil (threshold)
prediction = np.where(prediction >= 0.5, 1, 0)
prediction = [create_filled_convex_hull(k) for k in prediction]

# Calculer la Dice Loss
dice_loss = DiceLoss()
dice_scores = []
for i in range(image.shape[0]):
    dice = dice_loss(torch.from_numpy(prediction[i]), mask[i])
    dice_scores.append(dice.item())

# Afficher chaque image du batch et leur traitement
batch_size = image.shape[0]
fig, axs = plt.subplots(batch_size, 3, figsize=(12, 4*batch_size))
for i in range(batch_size):
    axs[i, 0].imshow(image[i].squeeze().cpu().detach().numpy(), cmap="gray")
    axs[i, 0].set_title("Image d'entrée")
    axs[i, 1].imshow(mask[i].squeeze().cpu().detach().numpy(), cmap="gray")
    axs[i, 1].set_title("Masque réel")
    axs[i, 2].imshow(prediction[i], cmap="gray")
    axs[i, 2].set_title("Prédiction")

plt.tight_layout()
plt.show()

# Calculer la moyenne de la Dice Loss sur le batch
mean_dice_score = np.mean(dice_scores)

print("Dice Loss (moyenne sur le batch) :", mean_dice_score)

"""**Test of the model on unlabelled dataset**"""

#load the unlabelled dataset with generator
PATH3 = '/content/drive/MyDrive/IIIA-CSIC/data/images/out_batch_3/roi_images/'
output_dir= '/content/drive/MyDrive/IIIA-CSIC/data/images/final_segmentation/'
TEST_IMAGE_PATH = os.path.join(PATH2, 'roi_images')
TEST_MASK_PATH = os.path.join(PATH2, 'roi_images')

train_generator,val_generator,test_generator = create_generators(PATH3,PATH3,mask_inversion=True,pretrain=False,data_augm=False,train_ratio=1.0, val_ratio=0.0, test_ratio=0.0)

itera=iter(train_generator)

#initialize model
model = AttentionUNet()
model.load_state_dict(torch.load('/content/drive/MyDrive/IIIA-CSIC/weights/outter_shape_aware_finetuning_weights_unet.pth',map_location=torch.device('cpu')))

batch_size=16

PATH3 = '/content/drive/MyDrive/IIIA-CSIC/data/images/out_batch_3/roi_images/'

# Transformation pour convertir les images en tensors et les normaliser
transform = transforms.Compose([
    transforms.ToTensor(),
])

# List of all images name
image_files = os.listdir(PATH3)

# We apply inference on all the images
k=0
for image_file in image_files:
    k+=1
    print('image '+k+' done')
    image_path = os.path.join(PATH3, image_file)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)/255
    #image=smart_resize(image,(192,192))

    image_tensor = transform(image).unsqueeze(0).to(torch.float)
    image_tensor=contrast(image_tensor,3)
    image_tensor = torch.clamp(image_tensor,0,1)

    with torch.no_grad():
        prediction = model(image_tensor)
        prediction = prediction.squeeze().cpu().detach().numpy()

    prediction = np.where(prediction >= 0.5, 255, 0).astype(np.uint8)

    fig, axs = plt.subplots(1, 2 , figsize=(7, 5))
    for i in range(batch_size):
        axs[0].imshow(image,cmap="gray")
        axs[0].set_title("Input")
        axs[1].imshow(prediction, cmap="gray")
        axs[1].set_title("Output")

    plt.tight_layout()
    # Save the plot as a PNG image
    #plt.savefig('/content/drive/MyDrive/IIIA-CSIC/data/images/final_segmentation_sample/'+str(image_file))
    plt.show()

    # To save the prediction in a file.
    # output_filename = image_file
    #output_path = os.path.join(output_dir, output_filename)
    #cv2.imwrite(output_path, prediction)
