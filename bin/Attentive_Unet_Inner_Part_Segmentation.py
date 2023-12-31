
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
from google.colab import drive
#drive.mount('/content/drive', force_remount=True) if using google collaboratory

#we define here the paths to access the data, first PATH is for the pretraining data, PATH2 is for the fine-tuning data
#You can change the paths here to adapt to your image location

PATH = 'drive/MyDrive/IIIA-CSIC/data/images/external_dataset/'
IMAGE_PATH = os.path.join(PATH, 'input')
MASK_PATH = os.path.join(PATH, 'mask')

PATH2 = 'drive/MyDrive/IIIA-CSIC/data/images/out_batch'
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
    Replace the black pixels of an image by the color given as argument
    '''
    black_pixels = (image == 0)
    image[black_pixels] = color

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

        #resizing + normal
        image = torch.tensor(smart_resize(cv2.imread(image_path),(self.image_size,self.image_size))[:,:,0])/255
        mask = torch.tensor(smart_resize(cv2.imread(mask_path),(self.image_size,self.image_size))[:,:,0])/255
        mask = torch.tensor(np.where(mask >= 0.5, 1, 0))

        #data_augmentation or not
        if self.data_augm:
          val = random.uniform(1, 3)
          image=contrast(image,val)
          im_bis=copy.copy(image)
          random_number = random.randint(0, 360)
          transform = T.RandomRotation((random_number,random_number),interpolation=InterpolationMode.BILINEAR,fill=image[0,0].item())
          if self.mask_inversion:
            transform = T.RandomRotation((random_number,random_number),interpolation=InterpolationMode.BILINEAR,fill=mask[0,0].item())

          image = transform(image.unsqueeze(0)).squeeze(0)
          image = replace_black_pixels(image, im_bis[0,0])

          mask =  transform(mask.unsqueeze(0)).squeeze(0)

        if self.mask_inversion:
            mask = torch.tensor(np.where(mask >= 0.5, 1, 0))


        e=mask.numpy()
        if self.pretrain:
          a=create_filled_convex_hull(e)[:,:,0]
          mask=torch.tensor(create_filled_convex_hull(e)[:,:,0]-e)


        image=torch.clamp(image,0,1)
        # Process and normalize data as needed
        image = image.clone().detach()
        mask = mask.clone().detach()

        X = image.unsqueeze(0)
        Y = mask.unsqueeze(0)

        return X, Y

def create_generators(image_folder, mask_folder,mask_inversion=False,pretrain=True,train_ratio = 0.7,val_ratio = 0.2,test_ratio = 0.1,data_augm=True):
    image_names = os.listdir(image_folder)
    mask_names = os.listdir(mask_folder)
    common_names = list(set(image_names) & set(mask_names))

    total_names = common_names  # Les noms de fichiers que vous souhaitez diviser
    total_count = len(total_names)

    slice1= round(total_count * train_ratio)
    slice2 = round(total_count * (val_ratio+train_ratio))


    train_names = total_names[:slice1]
    val_names = total_names[slice1:slice2]
    test_names = total_names[slice2:]
    print("val_names:",len(val_names))
    print("test_names",len(test_names))
    print("total count",total_count)
    print(val_names)
    print(test_names)
    train_generator = DataGenerator(image_folder, mask_folder, train_names,mask_inversion,pretrain,data_augm)
    val_generator = DataGenerator(image_folder, mask_folder, val_names,mask_inversion,pretrain,data_augm)
    test_generator = DataGenerator(image_folder, mask_folder, test_names,mask_inversion,pretrain,data_augm)



    train_loader = DataLoader(train_generator, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_generator, batch_size=VAL_BATCH_SIZE, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_generator, batch_size=TEST_BATCH_SIZE, shuffle=False, drop_last=True)

    return train_loader, val_loader, test_loader

# Initialization of the generators
train_loader, val_loader, test_loader = create_generators(TUNE_IMAGE_PATH, TUNE_MASK_PATH, train_ratio=0.7, val_ratio=0.3, test_ratio=0.0)

"""# Show batch"""

def create_filled_convex_hull(mask):
    '''
    function to make the mask convex and fill holes
    '''
    indices = np.argwhere(mask == 1)
    hull = ConvexHull(indices)
    image = np.zeros((192,192, 1), dtype=np.uint8) #416 ou 192 ici
    contour = indices[hull.vertices]

    # Inverser les coordonnées x et y du contour
    contour[:, [0, 1]] = contour[:, [1, 0]]

    # Remplir le contour avec une couleur spécifiée (blanc ici)
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

#we define some generators
train_loader,val_generator,test_generator = create_generators(TUNE_IMAGE_PATH, TUNE_MASK_PATH,mask_inversion=True,pretrain=False, train_ratio=0.8, val_ratio=0.2, test_ratio=0.0)

#Here we plot some of the images and masks

itera = iter(train_loader)
batch = next(itera)
X, Y = batch

for i in range(len(X)):
    image = (X[i]).squeeze(0).numpy()
    mask = (Y[i]).squeeze(0).numpy()

    mask_copy = copy.copy(mask)

    plt.figure(figsize=(10, 8))

    # Afficher le masque avec le contour échantillonné
    plt.subplot(2, 2, 3)
    plt.imshow(mask)
    plt.title("Mask")
    plt.axis('off')

    # Afficher l'image avec le contour échantillonné
    plt.subplot(2, 2, 4)
    plt.imshow(image)
    plt.title("Image")
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

#Here we define another Unet, this time with a spatial attention mechanism

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

# We define here the classical segmentation Dice Loss
class DiceLoss(nn.Module):
    def __init__(self, weight = None, size_average = True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth = 1.):


        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice

def resample_contour(contour, num_points):
    contour_length = contour.shape[0]
    indices = np.linspace(0, contour_length - 1, num_points, dtype=int)
    resampled_contour = contour[indices]
    return resampled_contour

class ShapeAwareLoss(nn.Module):
    def __init__(self, num_points=30):
        super(ShapeAwareLoss, self).__init__()
        self.num_points = num_points

    def forward(self, pred_mask, target_mask):
        # Compute binary cross entropy loss
        bce_loss = F.binary_cross_entropy_with_logits(pred_mask, target_mask, reduction='mean')

        # Convert predicted mask and target mask to numpy arrays
        pred_mask_np = pred_mask.detach().cpu().numpy()
        target_mask_np = target_mask.detach().cpu().numpy()

        # Apply sigmoid activation and convert to binary images
        pred_mask_binary = torch.sigmoid(pred_mask) > 0.5
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

            # Select the largest contour from each list
            pred_contour = max(pred_contours, key=cv2.contourArea)
            target_contour = max(target_contours, key=cv2.contourArea)

            # Resample the contours to have the same number of points
            pred_contour_resampled = resample_contour(pred_contour, self.num_points)
            target_contour_resampled = resample_contour(target_contour, self.num_points)

            # Convert contours to NumPy arrays
            pred_contour_np = np.array(pred_contour_resampled)
            target_contour_np = np.array(target_contour_resampled)

            # Compute Euclidean distance between the predicted and target contours
            distance = np.linalg.norm(pred_contour_np - target_contour_np)

            # Accumulate the batch loss
            batch_loss += bce_loss * distance

        # Compute the average loss over the batch
        shape_aware_loss = batch_loss / pred_mask.size(0)

        return shape_aware_loss

# Pretraining part


NB_EPOCHS=100
seed = 123
torch.manual_seed(seed)

# We initialize the model
model = AttentionUNet()
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")

# We initialize the generator
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

    #Choice of the loss function
    #criterion = DiceLoss()
    criterion = ShapeAwareLoss(num_points=30)
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

#launch the training
train()

# Save the weights of the pretrained model
torch.save(model.state_dict(),'/content/drive/MyDrive/IIIA-CSIC/weights/pretraining_weights_unet.pth')

#generator for plotting some results (test_generator if there is enoug images)
itera=iter(val_generator)

# Plot some predicions

image, mask = next(itera)
image = image.to('cuda')
model.to('cuda')
prediction = model(image).to('cuda')
pred2=copy.copy(prediction)
prediction = prediction.squeeze().cpu().detach().numpy()[0]

# Apply threshold
prediction = np.where(prediction >= 0.5, 1, 0)

# Plot image, mask a,d prediction
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

"""**Fine Tuning**"""

# Finetuning part
# Load the saved pretraining weights
model = AttentionUNet()
model.load_state_dict(torch.load(PRETRAINING_WEIGHTS_PATH,map_location=torch.device('gpu'))) #change to 'cpu' if you are using 'cpu' for inference

#Finetuning function

NB_EPOCHS=200
torch.manual_seed(112)

train_generator,val_generator,test_generator = create_generators(TUNE_IMAGE_PATH, TUNE_MASK_PATH,mask_inversion=True,pretrain=False, train_ratio=0.8, val_ratio=0.2, test_ratio=0.0)

def train():
    # Use gpu or cpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print('device:', device)

    model.to(device)

    # Loss function choice
    #criterion=DiceLoss()
    criterion = ShapeAwareLoss(num_points=30)
    #criterion=ContourLoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

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

# Launch the training
train()

#Save the weights of the finetuned model
torch.save(model.state_dict(),FINETUNING_WEIGHTS_PATH)

#plot some results
image, mask = next(itera)

model.to('cuda')
image = image.to('cuda')
prediction = model(image).to('cuda')
prediction = prediction.squeeze().cpu().detach().numpy()

# Apply threshold
prediction = np.where(prediction >= 0.5, 1, 0)

# Calculate Dice Loss
dice_loss = DiceLoss()
dice_scores = []
percent_diff_scores = []
for i in range(image.shape[0]):
    dice = dice_loss(torch.from_numpy(prediction[i]), mask[i])
    dice_scores.append(dice.item())

    # Calculate percentage of difference between the predicted and target surface
    pred_surface = np.sum(prediction[i])
    true_surface = torch.sum(mask[i]).item()
    percent_diff = (pred_surface - true_surface) / true_surface * 100
    percent_diff_scores.append(percent_diff)

batch_size = image.shape[0]
fig, axs = plt.subplots(batch_size, 3, figsize=(12, 4*batch_size))
for i in range(batch_size):
    im5=image[i].squeeze().cpu().detach().numpy()
    axs[i, 0].imshow(image[i].squeeze().cpu().detach().numpy(), cmap="gray")
    axs[i, 0].set_title("Image d'entrée")
    axs[i, 1].imshow(mask[i].squeeze().cpu().detach().numpy(), cmap="gray")
    axs[i, 1].set_title("Masque réel")
    axs[i, 2].imshow(prediction[i], cmap="gray")
    axs[i, 2].set_title("Prédiction")


plt.tight_layout()
plt.show()

# Calculate percentage of difference and dice loss for the image of the batch
mean_dice_score = np.mean(dice_scores)
mean_percent_diff = np.mean(percent_diff_scores)
std_percent_diff=np.std(percent_diff_scores)

print("Dice Loss (mean on the batch) :", mean_dice_score)
print("Pourcentage of difference of surface (mean on the batch) :", mean_percent_diff)
print("Standard deviation of percentage diff",std_percent_diff)

"""**Convex Correction**"""

itera=iter(val_generator)

#code to see the results, but with convex correction as post-processing
image, mask = next(itera)

image = image.to('cuda')
prediction = model(image)
prediction = prediction.squeeze().cpu().detach().numpy()

prediction = np.where(prediction >= 0.5, 1, 0)
prediction = [create_filled_convex_hull(k) for k in prediction]

dice_loss = DiceLoss()
dice_scores = []
for i in range(image.shape[0]):
    dice = dice_loss(torch.from_numpy(prediction[i]), mask[i])
    dice_scores.append(dice.item())

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

mean_dice_score = np.mean(dice_scores)

print("Dice Loss (mean on the batch) :", mean_dice_score)

"""**Test on unlabelled dataset**"""

# We now test our model on the unlabelled dataset

PATH3 = '/content/drive/MyDrive/IIIA-CSIC/data/images/out_batch_3/roi_images/' #path of the dataset
output_dir= '/content/drive/MyDrive/IIIA-CSIC/data/images/final_segmentation/' #path where the images will be saved
TEST_IMAGE_PATH = os.path.join(PATH2, 'roi_images')
TEST_MASK_PATH = os.path.join(PATH2, 'roi_images')

# Initialization og generators
train_generator,val_generator,test_generator = create_generators(PATH3,PATH3,mask_inversion=True,pretrain=False,data_augm=False,train_ratio=1.0, val_ratio=0.0, test_ratio=0.0)

itera=iter(train_generator)

model = AttentionUNet()
model.load_state_dict(torch.load('/content/drive/MyDrive/IIIA-CSIC/weights/shape_aware_finetuning_weights_unet.pth',map_location=torch.device('cpu')))

batch_size=16
# Supposons que vous avez déjà défini les variables suivantes :
# model : modèle utilisé pour la segmentation
# output_dir : répertoire de sortie pour enregistrer les prédictions
PATH3 = '/content/drive/MyDrive/IIIA-CSIC/data/images/out_batch_3/roi_images/'

# Transformation pour convertir les images en tensors et les normaliser
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Liste pour stocker les noms des fichiers d'images
image_files = os.listdir(PATH3)

# Parcourir toutes les images dans le dossier
k=0
for image_file in image_files:
    k+=1
    print('image k done')
    # Chemin complet de l'image d'entrée
    image_path = os.path.join(PATH3, image_file)

    # Charger l'image en utilisant OpenCV (cv2)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)/255
    #image=smart_resize(image,(192,192))

    # Appliquer la transformation à l'image
    image_tensor = transform(image).unsqueeze(0).to(torch.float)
    image_tensor=contrast(image_tensor,3)
    image_tensor = torch.clamp(image_tensor,0,1)   #.to('cuda')

    # Passer l'image à travers le modèle pour obtenir la prédiction
    with torch.no_grad():
        #model.to('cuda')
        prediction = model(image_tensor)
        prediction = prediction.squeeze().cpu().detach().numpy()

    # Appliquer le seuil (threshold)
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

    # output_filename = image_file
    #output_path = os.path.join(output_dir, output_filename)
    #cv2.imwrite(output_path, prediction)
