# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 08:22:49 2022

@author: Usuario
"""

import os
#os.environ['KMP_DUPLICATE_LIB_OK']='True'
import os
import shutil
import tempfile
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
)

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import UNETR

from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)

### training and validation dataset
#path_train_volumes
###### create nnunet monaie for multiclass 3D segmentation problem
import numpy as np
from monai.transforms import (
    Compose,
    LoadImaged,
    AddChanneld,
    CropForegroundd,
    Spacingd,
    Orientationd,
    SpatialPadd,
    NormalizeIntensityd,
    RandCropByPosNegLabeld,
    RandRotated,
    RandZoomd,
    CastToTyped,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandAdjustContrastd,
    RandFlipd,
    ToTensord,
    Resized,RandShiftIntensityd,SpatialPadd
)
from monai.utils import first

### tranform the images
import os
from glob import glob
import shutil
from tqdm import tqdm
#import dicom2nifti
import numpy as np
import nibabel as nib
from monai.transforms import(
    Compose,
    AddChanneld,
    LoadImaged,
    Resized,
    ToTensord,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
)
from monai.data import DataLoader, Dataset, CacheDataset
from monai.utils import set_determinism
set_determinism(seed=0)
import os
import nibabel as nib
import glob
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
    Invertd,KeepLargestConnectedComponent,
)


data_dir='C:\\Users\\Usuario\\Desktop\\Data\\Motiondataset\\CMRxMotion Challenge Validation Data\\CMRxMotion_validation'

test_images = sorted(glob.glob(os.path.join(data_dir, "*", "*.nii.gz")))


test_data = [{"image": image} for image in test_images]


test_org_transforms = Compose(
    [
        LoadImaged(keys="image"),
        EnsureChannelFirstd(keys="image"),
        #Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(
            1.5, 1.5, 2.0), mode="bilinear"),
        ScaleIntensityRanged(
            keys=["image"], a_min=-700, a_max=300,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        CropForegroundd(keys=["image"], source_key="image"),
        EnsureTyped(keys="image"),
    ]
)

# val_transforms = Compose(
#     [
#         LoadImaged(keys=["images", "labels"]),
#         EnsureChannelFirstd(keys=["images", "labels"]),
#         Orientationd(keys=["images", "labels"], axcodes="LPS"),
#         Spacingd(keys=["images", "labels"], pixdim=(
#             1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
#         ScaleIntensityRanged(
#             keys=["images"], a_min=-700, a_max=300,
#             b_min=0.0, b_max=1.0, clip=True,
#         ),
#         CropForegroundd(keys=["images", "labels"], source_key="images"),
#         EnsureTyped(keys=["images", "labels"]),
#     ]
# )

test_org_ds = Dataset(
    data=test_data, transform=test_org_transforms)

test_org_loader = DataLoader(test_org_ds, batch_size=1, num_workers=0)

print(test_org_loader.dataset.data[0]['image'])
#dir(test_org_loader.dataset.data['image'])

post_transforms = Compose([
    EnsureTyped(keys="pred"),
    Invertd(
        keys="pred",
        transform=test_org_transforms,
        orig_keys="image",
        meta_keys="pred_meta_dict",
        orig_meta_keys="image_meta_dict",
        meta_key_postfix="meta_dict",
        nearest_interp=False,
        to_tensor=True,
    ),
    AsDiscreted(keys="pred", argmax=True, to_onehot=4),
    SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir="./out", output_postfix="seg", resample=False),
])

#%% model evlaution
from monai.networks.nets import UNet
from monai.networks.layers import Norm

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model=get_DynUNet(in_channels, n_class, device)
root_dir='C:\\Users\\Usuario\\Desktop\\Data\\Motiondataset\\model1'
model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))
model.eval()

#%% testing function
pathsave='C:\\Users\\Usuario\\Desktop\\Data\\Motiondataset\\save_predictionsnew'
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
        
import SimpleITK as sitk
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
def get_largest_component(image):
    """
    get the largest component from 2D or 3D binary image
    image: nd array
    """
    dim = len(image.shape)
    if(image.sum() == 0 ):
        print('the largest component is null')
        return image
    if(dim == 2):
        s = ndimage.generate_binary_structure(2,1)
    elif(dim == 3):
        s = ndimage.generate_binary_structure(3,1)
    else:
        raise ValueError("the dimension number should be 2 or 3")
    labeled_array, numpatches = ndimage.label(image, s)
    sizes = ndimage.sum(image, labeled_array, range(1, numpatches + 1))
    max_label = np.where(sizes == sizes.max())[0] + 1
    output = np.asarray(labeled_array == max_label, np.uint8)
    return  output

#cc = sitk.ConnectedComponent(segmentation,True)


def largestcc(segmentation):
    #segmentation= sitk.ReadImage(segmentation_file_path)
    # print(type(segmentation))

    cc = sitk.ConnectedComponent(segmentation,True)

    stats = sitk.LabelIntensityStatisticsImageFilter()
    stats.Execute(cc,segmentation)

    largestCClabel = 0
    largestCCsize = 0 

    for l in stats.GetLabels():
        #print("Label: {0} -> Mean: {1} Size: {2}".format(l, stats.GetMean(l), int(stats.GetPhysicalSize(l))))
        if int(stats.GetPhysicalSize(l)) >= largestCCsize:
            largestCCsize = int(stats.GetPhysicalSize(l))
            largestCClabel = l
    
    largestCC = cc == largestCClabel # get the largest component
    return largestCC


# argmax = [AsDiscrete(argmax=True)(i) for i in decollate_batch(val_output)]
#         plt.imshow(argmax[0].detach().cpu()[0, :, :, 80])
#         plt.subplot(1, 5, 3)
#         plt.title(f"largest {i}")
#         largest = [KeepLargestConnectedComponent(applied_labels=[1])(i) for i in argmax]

import numpy as np
from skimage.measure import label

def getLargestCC(segmentation):
    labels = label(segmentation)
    unique, counts = np.unique(labels, return_counts=True)
    list_seg=list(zip(unique, counts))[1:] # the 0 label is by default background so take the rest
    largest=max(list_seg, key=lambda x:x[1])[0]
    labels_max=(labels == largest).astype(int)
    return labels_max

import random
import warnings
from typing import Optional, Callable

import torch
import numpy as np

from monai.config import IndexSelection
from monai.utils import ensure_tuple, ensure_tuple_size, fall_back_tuple, optional_import, min_version

measure, _ = optional_import("skimage.measure", "0.14.2", min_version)

def get_largest_connected_component_mask(img, connectivity: Optional[int] = None):
    """
    Gets the largest connected component mask of an image.

    Args:
        img: Image to get largest connected component from. Shape is (batch_size, spatial_dim1 [, spatial_dim2, ...])
        connectivity: Maximum number of orthogonal hops to consider a pixel/voxel as a neighbor.
            Accepted values are ranging from  1 to input.ndim. If ``None``, a full
            connectivity of ``input.ndim`` is used.
    """
    img_arr = img.detach().cpu().numpy()
    largest_cc = np.zeros(shape=img_arr.shape, dtype=img_arr.dtype)
    for i, item in enumerate(img_arr):
        item = measure.label(item, connectivity=connectivity)
        if item.max() != 0:
            largest_cc[i, ...] = item == (np.argmax(np.bincount(item.flat)[1:]) + 1)
    return torch.as_tensor(largest_cc, device=img.device) 

#labels_in = np.ones((512, 512, 512), dtype=np.int32)
#labels_out = cc3d.connected_components(labels_in) # 26-connected
import cc3d
import numpy as np
#https://github.com/seung-lab/connected-components-3d
#labels_out, N = cc3d.connected_components(labels_in, return_N=True) # free
# -- OR -- 
#labels_out = cc3d.connected_components(labels_in) 
#N = np.max(labels_out) # costs a full read
with torch.no_grad():
    for i,test_data in enumerate(test_org_loader):
        test_inputs = test_data["image"].to(device)
        sub=test_org_loader.dataset.data[i]['image']
        sub_p=sub.split('\\')[-2]
        file_p=sub.split('\\')[-1]
        print(test_inputs.shape)
        roi_size = (160,160,160)
        sw_batch_size = 1
        test_output= sliding_window_inference(test_inputs, roi_size, sw_batch_size, model,0.8)
        prediction=torch.argmax(test_output, dim=1).detach().cpu()
        #cc=get_largest_connected_component_mask(prediction)
        prdic_numpy=prediction.squeeze(axis=0).numpy()
        #connectivity = 6
        #prdic_numpy = cc3d.connected_components(prdic_numpy,connectivity=connectivity)
        #prdic_numpy = cc3d.connected_components(prdic_numpy) 
        #prdic_numpy, N = cc3d.largest_k(prdic_numpy, k=3,connectivity=26, delta=0,
        #                               return_N=True,)
        #N = np.max(labels_out) # costs a full read
        #print(labels_out.shape)
        
        #break
        
        #prediction=torch.argmax(test_output, dim=1).detach().cpu()
        
        #print(prediction.shape)
        #prdic_numpy=labels_out.squeeze(axis=0).numpy()
        print(prdic_numpy.shape)
        print(np.unique(prdic_numpy))
        #break
        #prdic_numpy=getLargestCC(prdic_numpy)
        #largest = [KeepLargestConnectedComponent(applied_labels=[1])(i) for i in prdic_numpy]
        #prdic_numpy=largest[0]
        #prdic_numpy=np.swapaxes(prdic_numpy, 2, 0)
        #cc = sitk.ConnectedComponent(prdic_numpy,True)
        #prdic_numpy=get_largest_component(prdic_numpy)
        #segmentation=sitk.GetImageFromArray(prdic_numpy)
        #cc=largestcc(segmentation)
        #prdic_numpy=sitk.GetArrayFromImage(cc)
        prdic_numpy=np.swapaxes(prdic_numpy, 2, 0)
        
        predicted_volume = sitk.GetImageFromArray(prdic_numpy, isVector=False)
        #print("Size of segmented volume: ", predicted_volume.GetSize())
        p=os.path.join(pathsave, sub_p)
        #createFolder(p)
        #p="train_accu53new1.nii.gz"
        sitk.WriteImage(sitk.Cast( predicted_volume, sitk.sitkUInt8 ), os.path.join(pathsave,file_p), True)
        
        #break

        #test_data = [post_transforms(i) for i in decollate_batch(test_data)]
# #%% new test function
# pathsave='C:\\Users\\Usuario\\Desktop\\Data\\Motiondataset\\save_predictions'
# def createFolder(directory):
#     try:
#         if not os.path.exists(directory):
#             os.makedirs(directory)
#     except OSError:
#         print ('Error: Creating directory. ' +  directory)
  
# post_transforms = Compose([
#     EnsureTyped(keys="pred"),
#     Invertd(
#         keys="pred",
#         transform=test_org_transforms,
#         orig_keys="image",
#         meta_keys="pred_meta_dict",
#         orig_meta_keys="image_meta_dict",
#         meta_key_postfix="meta_dict",
#         nearest_interp=False,
#         to_tensor=True,
#     ),
#     AsDiscreted(keys="pred", argmax=True, to_onehot=4),
#     SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir=pathsave, output_postfix="seg", resample=False),
# ])

# import SimpleITK as sitk
# with torch.no_grad():
#     for i,test_data in enumerate(test_org_loader):
#         test_inputs = test_data["image"].to(device)
#         sub=test_org_loader.dataset.data[0]['image']
#         sub_p=sub.split('\\')[-2]
#         file_p=sub.split('\\')[-1]
#         #print(test_inputs.shape)
#         roi_size = (128,128,64)
#         sw_batch_size = 4
#         test_data["pred"]= sliding_window_inference(test_inputs, roi_size, sw_batch_size, model)
#         test_data = [post_transforms(i) for i in decollate_batch(test_data)]
#         #prediction=torch.argmax(test_output, dim=1).detach().cpu()
#         #test_data["pred"]
#         # #print(prediction.shape)
#         # prdic_numpy=prediction.squeeze(axis=0).numpy()
#         # print(prdic_numpy.shape)
#         # print(np.unique(prdic_numpy))
#         # prdic_numpy=np.swapaxes(prdic_numpy, 2, 0)
        
#         # predicted_volume = sitk.GetImageFromArray(prdic_numpy, isVector=False)
#         # #print("Size of segmented volume: ", predicted_volume.GetSize())
#         # p=os.path.join(pathsave, sub_p)
#         # createFolder(p)
#         # #p="train_accu53new1.nii.gz"
#         # sitk.WriteImage(sitk.Cast( predicted_volume, sitk.sitkUInt8 ), os.path.join(p,file_p), True)
        
#         break
     
#%% test images
#path='C:\Users\Usuario\Desktop\Data\Motiondataset\CMRxMotion Challenge Validation Data\CMRxMotion_validation\P022-1'