# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 10:12:10 2022

@author: Administrateur
"""

#%% Monaie segmentation using parse2022
from monai.utils import first, set_determinism
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd, 
    EnsureTyped,
    EnsureType,
    Invertd
)

#from monai.handlers.utils import from_engine
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract
import aim
import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob

import nibabel as nib
import numpy as np
from tqdm import tqdm

# os.makedirs("./data")
root_dir = "/home/imranr/monabdul/Parse2022/train"

train_images = sorted(glob.glob(os.path.join(root_dir, "*", 'image', "*.nii.gz")))
train_labels = sorted(glob.glob(os.path.join(root_dir, "*", 'label', "*.nii.gz")))

data_dicts = [{"images": images_name, "labels": label_name} for images_name, label_name in zip(train_images, train_labels)]
train_files, val_files = data_dicts[:-9], data_dicts[-9:]

set_determinism(seed = 0)

train_transforms = Compose(
    [
     LoadImaged(keys=['images', 'labels']),
     EnsureChannelFirstd(keys = ["images", "labels"]),
     Orientationd(keys=['images', 'labels'], axcodes = 'LPS'),
     Spacingd(keys=['images', 'labels'], pixdim = (1.5,1.5,2), mode = ("bilinear", 'nearest')),
     ScaleIntensityRanged(
            keys=["images"], a_min=-700, a_max=300,
            b_min=0.0, b_max=1.0, clip=True,
        ),
     CropForegroundd(keys=['images', 'labels'], source_key="images"),
     RandCropByPosNegLabeld(
            keys=['images', 'labels'],
            label_key="labels",
            spatial_size=(96, 96, 96),
            pos=1,
            neg=1,
            num_samples=4,
            image_key="images",
            image_threshold=0,
        ),
     EnsureTyped(keys=['images', 'labels']),     
          
    ]
)

val_transforms = Compose(
    [
        LoadImaged(keys=["images", "labels"]),
        EnsureChannelFirstd(keys=["images", "labels"]),
        Orientationd(keys=["images", "labels"], axcodes="LPS"),
        Spacingd(keys=["images", "labels"], pixdim=(
            1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(
            keys=["images"], a_min=-700, a_max=300,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        CropForegroundd(keys=["images", "labels"], source_key="images"),
        EnsureTyped(keys=["images", "labels"]),
    ]
)

check_ds = Dataset(data = val_files, transform = val_transforms)
check_loader = DataLoader(check_ds, batch_size = 1)
check_data = first(check_loader)
img, label = check_data['images'], check_data['labels']

check_ds = Dataset(data = train_files, transform = train_transforms)
check_loader = DataLoader(check_ds, batch_size = 1)
check_data = first(check_loader)
img, label = check_data['images'], check_data['labels']

train_ds = CacheDataset(
    data = train_files, transform = train_transforms,
    cache_rate = 1.0, num_workers = 2
)

train_loader = DataLoader(train_ds, batch_size = 2, shuffle = True, num_workers=2)
val_ds = CacheDataset(
    data = val_files, transform = val_transforms,
    cache_rate = 1.0, num_workers = 2
)
val_loader = DataLoader(val_ds, batch_size = 1, shuffle = False, num_workers=2)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 
# model = UNet(spatial_dims=3,
#              in_channels=1, 
#              out_channels=2,
#              channels = (16,32,64,128,256),
#              strides = (2,2,2,2),
#              num_res_units = 2,
#              norm = Norm.BATCH
#              ).to(device)        

UNet_meatdata = dict(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH
)

model = UNet(**UNet_meatdata).to(device)

#model.load_state_dict(torch.load("/content/drive/MyDrive/best_metric_model_1.pth", map_location=torch.device('cpu')))
loss_function = DiceLoss(to_onehot_y=True, softmax=True)
loss_type = "DiceLoss"
optimizer = torch.optim.Adam(model.parameters(), 1e-4)
dice_metric = DiceMetric(include_background=False, reduction="mean")

Optimizer_metadata = {}
for ind, param_group in enumerate(optimizer.param_groups):
    optim_meta_keys = list(param_group.keys())
    Optimizer_metadata[f'param_group_{ind}'] = {key: value for (key, value) in param_group.items() if 'params' not in key}
    

max_epochs = 500
val_interval = 10
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []
post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=2)])
post_label = Compose([EnsureType(), AsDiscrete(to_onehot=2)])

# initialize a new Aim Run
aim_run = aim.Run()
#log model metadata
aim_run['UNet_meatdata'] = UNet_meatdata
#log optimizer metadata
aim_run['Optimizer_metadata'] = Optimizer_metadata

#root_dir=''
slice_to_track = 80

for epoch in tqdm(range(max_epochs)):
  model.train()
  epoch_loss = 0
  step = 0
  for batch_data in train_loader:
    step += 1
    inputs, labels = (
        batch_data['images'].to(device),
        batch_data['labels'].to(device)
    )
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = loss_function(outputs, labels)
    loss.backward()
    optimizer.step()
    epoch_loss +=loss.item()
    print(f"{step}/{len(train_ds) // train_loader.batch_size}, "
            f"train_loss: {loss.item():.4f}")
    aim_run.track(loss.item(), name="batch_loss", context={'type':loss_type})

  epoch_loss /= step
  epoch_loss_values.append(epoch_loss)
  aim_run.track(epoch_loss, name="epoch_loss", context={'type':loss_type})

  print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

  if (epoch + 1) % val_interval == 0:

    model.eval()
    with torch.no_grad():
      for index, val_data in enumerate(val_loader):
        val_inputs, val_labels = val_data['images'].to(device), val_data['labels'].to(device)
        roi_size = (160, 160, 160)
        sw_batch_size = 4
        val_outputs = sliding_window_inference(
                    val_inputs, roi_size, sw_batch_size, model)
        
        output = torch.argmax(val_outputs, dim=1)[0, :, :, slice_to_track].float()

        # aim_run.track(aim.Image(val_inputs[0, 0, :, :, slice_to_track], \
        #                                 caption=f'Input Image: {index}'), \
        #                        name='validation', context={'type':'input'})
        # aim_run.track(aim.Image(val_labels[0, 0, :, :, slice_to_track], \
        #                         caption=f'Label Image: {index}'), \
        #                 name='validation', context={'type':'label'})
        # aim_run.track(aim.Image(output, caption=f'Predicted Label: {index}'), \
        #                 name = 'predictions', context={'type':'labels'})
                      
        val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
        val_labels = [post_label(i) for i in decollate_batch(val_labels)]
        dice_metric(y_pred=val_outputs, y=val_labels)

      metric = dice_metric.aggregate().item()
      aim_run.track(metric, name="val_metric", context={'type':loss_type})
      dice_metric.reset()

      metric_values.append(metric)
      if metric > best_metric:
        best_metric = metric
        best_metric_epoch = epoch + 1
        torch.save(model.state_dict(), os.path.join(
            root_dir, "best_metric_model_parse2022.pth"))
        
        best_model_log_message = f"saved new best metric model at the {epoch+1}th epoch"
        aim_run.track(aim.Text(best_model_log_message), name='best_model_log_message', epoch=epoch+1)
        print(best_model_log_message)
              
        message1 = f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
        message2 = f"\nbest mean dice: {best_metric:.4f} "
        message3 = f"at epoch: {best_metric_epoch}"
  
        aim_run.track(aim.Text(message1 +"\n" + message2 + message3), name='epoch_summary', epoch=epoch+1)
        print(message1, message2, message3)
        
        
