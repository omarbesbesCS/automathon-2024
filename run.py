#!/usr/bin/env python3
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary
import torchvision.io as io
import os
import json
from tqdm import tqdm
import csv
import timm
import wandb
import torch.nn.functional as F
import numpy as np


from PIL import Image
import torchvision.transforms as transforms

# UTILITIES

import matplotlib.pyplot as plt

def display_image(img):
    # img = img.permute(2,1,0)
    print(img.shape)
    plt.imshow(img)

def smart_resize(data, size): # kudos louis
    # Prends un tensor de shape [...,C,H,W] et le resize en [C,new_height,new_width]
    # x, y, height et width servent a faire un crop avant de resize
    full_height = data.shape[-2]
    full_width = data.shape[-1]

    if full_height > full_width:
        alt_height = size
        alt_width = int(full_width * full_height / size)
    elif full_height < full_width:
        alt_height = int(full_height * full_width / size)
        alt_width = size
    else:
        alt_height = size
        alt_width = size
    tr = transforms.Compose([
        transforms.Resize((alt_height, alt_width)),
        transforms.CenterCrop(size)
    ])
    print(data.shape)
    return tr(data)

def resize_image(image_tensor, target_size):
    image_tensor_batched = image_tensor
    resized_image_batched = F.interpolate(image_tensor_batched.permute(2, 0, 1).unsqueeze(0), size=target_size, mode='bilinear', align_corners=False)
    resized_image = resized_image_batched.squeeze(0).permute(1,2,0)
    return resized_image

def resize_data(data, new_height, new_width, x=0, y=0, height=None, width=None):
    # Prends un tensor de shape [...,C,H,W] et le resize en [C,new_height,new_width]
    # x, y, height et width servent a faire un crop avant de resize
    
    full_height = data.shape[-2]
    full_width = data.shape[-1]
    height = full_height - y if height is None else height
    width = full_width -x if width is None else width

    ratio = new_height/new_width
    if height/width > ratio:
        expand_height = height
        expand_width = int(height / ratio)
    elif height/width < ratio:
        expand_height = int(width * ratio)
        expand_width = width
    else:
        expand_height = height
        expand_width = width
    tr = transforms.Compose([
        transforms.CenterCrop((expand_height, expand_width)),
        transforms.Resize((new_height, new_width))
    ])
    x = data[...,y:min(y+height, full_height), x:min(x+width, full_width)].clone()
    return tr(x)
    
# SETUP DATASET

dataset_dir = "./datasets/"
root_dir = os.path.expanduser("./datasets/")

nb_frames = 30

class VideoDataset(Dataset):
    """
    This Dataset takes a video and returns a tensor of shape [10, 3, 256, 256]
    That is 10 colored frames of 256x256 pixels.
    """
    def __init__(self, root_dir, dataset_choice="train", nb_frames=10):
        super().__init__()
        self.dataset_choice = dataset_choice
        if  self.dataset_choice == "train":
            self.root_dir = os.path.join(root_dir, "train_dataset")
        elif  self.dataset_choice == "test":
            self.root_dir = os.path.join(root_dir, "test_dataset")
        elif  self.dataset_choice == "experimental":
            print(os.path.join(root_dir, "experimental_dataset"))
            self.root_dir = os.path.join(root_dir, "experimental_dataset")
        else:
            raise ValueError("choice must be 'train', 'test' or 'experimental'")

        with open(os.path.join(root_dir, "dataset.csv"), 'r') as file:
            reader = csv.reader(file)
            # read dataset.csv with id,label columns to create
            # a dict which associated label: id
            self.ids = {row[1][:-3] + "pt" : row[0] for row in reader}
            print(self.ids)
        if self.dataset_choice == "test":
            self.data = None
        else:
            with open(os.path.join(self.root_dir, "metadata.json"), 'r') as file:
                self.data= json.load(file)
                self.data = {k[:-3] + "pt" : (torch.tensor(float(1)) if v == 'fake' else torch.tensor(float(0))) for k, v in self.data.items()}

        #self.video_files = [f for f in os.listdir(self.root_dir) if f.endswith('.mp4')]
        #print("actccqfmac.mp4" in self.ids)
        self.video_files = [f for f in os.listdir(self.root_dir) if f.endswith('.pt')]

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = os.path.join(self.root_dir, self.video_files[idx])
        # print(video_path)
        video, audio, info = io.read_video(video_path, pts_unit='sec')
        # video = torch.load(video_path)
        '''
        video = video.permute(0,3,1,2)
        length = video.shape[0]
        video = video[[i*(length//(nb_frames)) for i in range(nb_frames)]]
        '''
        frames=[]
        k=0
        for i,frame in enumerate(video):
            if(k>=150):
                break
            if((k%5)!=0):
                k+=1
                continue
            k+=1
            frame2=resize_image(frame,(270,480))
            frame_np=frame2.numpy()
            frames.append(frame_np)
        # resize the data into a reglar shape of 256x256 and normalize it
        # video = smart_resize(video, 256) / 255
        #video = resize_image(video, (324,567)) / 255
        # video = video / 255
        video=torch.Tensor(np.array(frames))/255
        # print(self.video_files[idx])
        ID = self.ids[self.video_files[idx]]
        if self.dataset_choice == "test":
            return video, ID
        else:
            label = self.data[self.video_files[idx]]
            return video, label, ID    


train_dataset = VideoDataset(dataset_dir, dataset_choice="train", nb_frames=nb_frames)
test_dataset = VideoDataset(dataset_dir, dataset_choice="test", nb_frames=nb_frames)
#experimental_dataset = VideoDataset(dataset_dir, dataset_choice="experimental", nb_frames=nb_frames)


# MODELE

from torch.nn import LogSoftmax
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        """self.conv1 = nn.Conv3d(in_channels=3, out_channels=20,kernel_size=(5, 5))
        self.relu1 = nn.ReLU()
        #self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # initialize second set of CONV => RELU => POOL layers
        self.conv2 = nn.Conv3d(in_channels=20, out_channels=50,
        kernel_size=(5, 5))
        self.relu2 = nn.ReLU()
        #self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # initialize first (and only) set of FC => RELU layers
        self.fc1 = nn.Linear(in_features=800, out_features=500)
        self.relu3 = nn.ReLU()
        # initialize our softmax classifier
        self.fc2 = nn.Linear(in_features=500, out_features=2)
        self.sigmoid = nn.Sigmoid()"""
        self.conv1 = nn.Conv3d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(380160, 512) # Adjust this based on input image size //64 * 37 * 25
        self.fc2 = nn.Linear(512, 2) # 2 classes: cat and dog
        self.Softmax = nn.Softmax(dim=1)


    def forward(self, x):
        """x = self.conv1(x)
        x = self.relu1(x)
        # pass the output from the previous layer through the second
        # set of CONV => RELU => POOL layers
        x = self.conv2(x)
        x = self.relu2(x)
        # flatten the output from the previous layer and pass it
        # through our only set of FC => RELU layers
        x = flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        # pass the output to our softmax classifier to get our output
        # predictions
        x = self.fc2(x)
        output = self.sigmoid(x)
        # return the output predictions
        return output"""
        y = self.pool(nn.functional.relu(self.conv1(x)))
        y = self.pool(nn.functional.relu(self.conv2(y)))
        y = self.pool(nn.functional.relu(self.conv3(y)))
        y=self.flat(y)
        y = nn.functional.relu(self.fc1(y))
        y = self.fc2(y)
        y = self.Softmax(y)
        return y

# LOGGING

wandb.login(key="2032122fa7aa881297fbcc27805daa6f416e4667")

run = wandb.init(
    project="automathon",
    name="test-admin",
    config={
        "learning_rate": 0.001,
        "architecture": "-",
        "dataset": "DeepFake Detection Challenge",
        "epochs": 10,
        "batch_size": 10,
    },
)

# ENTRAINEMENT

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("mps")
loss_fn = nn.MSELoss()
model = CNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
nb_epochs = 5
loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

for epoch in range(nb_epochs):
    for sample in tqdm(loader, desc="Epoch {}".format(epoch), ncols=0):
        optimizer.zero_grad()
        X, label, ID = sample
        
        X=X.permute(0,4,1,2,3)
        #print(sample)
        """print(X.shape)
        print(label.shape)
        print(X)
        print("*************")
        print(label)
        """
        X = X
        label = label
        label_pred = model(X)
        label=torch.unsqueeze(label,dim=1)
        #print(label,label_pred.detach().numpy()[:,0:1])
        loss = loss_fn(label, torch.from_numpy(label_pred.detach().numpy()[:,0:1]))
        loss.requires_grad = True
        loss.backward()
        optimizer.step()
        print("loss ",loss.item())
        run.log({"loss": loss.item(), "epoch": epoch})


## TEST


loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
#model = model.to(device)
ids = []
labels = []
print("Testing...")
for sample in tqdm(loader):
    X, ID = sample
    X=X.permute(0,4,1,2,3)
    #ID = ID[0]
    #X = X.to(device)
    label_pred = model(X)
    ids.extend(list(ID))
    print(label_pred)
    pred = (label_pred > 0.5).long()
    pred = pred.cpu().detach().numpy().tolist()
    labels.extend(pred)



### ENREGISTREMENT
print("Saving...")
tests = ["id,label\n"] + [f"{ID},{label_pred[0]}\n" for ID, label_pred in zip(ids, labels)]
with open("submission.csv", "w") as file:
    file.writelines(tests)
