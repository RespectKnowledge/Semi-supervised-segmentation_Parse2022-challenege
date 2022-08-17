# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 16:47:37 2022

@author: Administrateur
"""
#%% cross validation of dataset
import os
import nibabel as nib
import glob
import SimpleITK as sitk
path='C:\\Users\\Administrateur\\Desktop\\mmchalleneges2022\\Parse2021\\train\\train'
#pathim=os.path.join(path,'image')
#pathlabel=os.path.join(path,'label')
lstdir=sorted(os.listdir(path))
#### get images path
#patients_pattern_img = os.path.join(path,'*','*image')
patients_images = glob.glob(os.path.join(path,'*','*image','*'))
###### get label paths
#img=sorted(glob.glob(os.path.join(lstdir,'*.nii.gz')))
#label=sorted(glob.glob(os.path.join(pathlabel,'*.nii.gz')))
#patients_pattern_label = os.path.join(path,'*','*label')
patients_label = glob.glob(os.path.join(path,'*','*label','*'))

import random
#random.seed(0)
def Trian_val(data_list,test_size=0.15):
    n=len(data_list)
    m=int(n*test_size)
    test_item=random.sample(data_list,m)
    train_item=list(set(data_list)-set(test_item))
    return train_item,test_item
tr_list,test_list=Trian_val(patients_images,test_size=0.20)

import pandas as pd
df_test= pd.DataFrame(test_list,columns=['PatientID'])
df_train= pd.DataFrame(tr_list,columns=['PatientID'])
# df.to_csv("testing_list1.csv",index=False)
# df1e=df11.drop(['index'],axis=0)
# df11.values()
df_test.to_csv('valid_fold1parse.csv', index=False) 
df_train.to_csv('train_fold1parse.csv', index=False)  

#%% load fold1 dataset
import os
import pandas as pd
import nibabel as nib
import SimpleITK as sitk
import numpy as np
path='C:\\Users\\Administrateur\\Desktop\\mmchalleneges2022\\Parse2021'
pathc=pd.read_csv(os.path.join(path,'train_fold1parse.csv'))
savetrain='C:\\Users\\Administrateur\\Desktop\\mmchalleneges2022\\Parse2021\\NNUnet_data_prepartion\\imagesTr'
savetrainlabel='C:\\Users\\Administrateur\\Desktop\\mmchalleneges2022\\Parse2021\\NNUnet_data_prepartion\\labelsTr'
#n_objct=nib.load(img[0]).get_data()
for i in range(0,len(pathc)):
    pathd=pathc['PatientID'].iloc[i]
    #print(pathd)
    pathm=pathd.replace('image','label')
    data=nib.load(pathd)
    mask=nib.load(pathm)
    #label1=nib.load(patients_label[i]).get_data()
    #print('image:',n_objct.shape)
    #print('label:',label1.shape)
    #imge=sitk.ReadImage(patients_images[i])
    #print(imge.GetSpacing())
    #print(np.unique(mask))
    print(pathd[72:80])
    nib.save(data, os.path.join(savetrain,pathd[72:80]+"_0000"+'.nii.gz'))
    nib.save(mask, os.path.join(savetrainlabel,pathd[72:80]+'.nii.gz'))
    #break
    
#%% testing dataset

import os
import pandas as pd
import nibabel as nib
import SimpleITK as sitk
import numpy as np
path='C:\\Users\\Administrateur\\Desktop\\mmchalleneges2022\\Parse2021'
pathc=pd.read_csv(os.path.join(path,'valid_fold1parse.csv'))
savetrain='C:\\Users\\Administrateur\\Desktop\\mmchalleneges2022\\Parse2021\\NNUnet_data_prepartion\\imagesTs'

savetrain_label='C:\\Users\\Administrateur\\Desktop\\mmchalleneges2022\\Parse2021\\NNUnet_data_prepartion\\labelsts'

#savetrainlabel='C:\\Users\\Administrateur\\Desktop\\mmchalleneges2022\\Parse2021\\NNUnet_data_prepartion\\labelsTr'
#n_objct=nib.load(img[0]).get_data()
for i in range(0,len(pathc)):
    pathd=pathc['PatientID'].iloc[i]
    #print(pathd)
    #pathm=pathd.replace('image','label')
    data=nib.load(pathd.replace('image','label'))
    #break
    #mask=nib.load(pathm)
    #label1=nib.load(patients_label[i]).get_data()
    #print('image:',n_objct.shape)
    #print('label:',label1.shape)
    #imge=sitk.ReadImage(patients_images[i])
    #print(imge.GetSpacing())
    #print(np.unique(mask))
    print(pathd[72:80])
    nib.save(data, os.path.join(savetrain_label,pathd[72:80]+'.nii.gz'))
    #nib.save(mask, os.path.join(savetrainlabel,pathd[72:80]+'.nii.gz'))
    #break
#%json file
#%
#%% create json for  dataset
import shutil
from collections import OrderedDict
import json
import numpy as np
import os

#visualization of the dataset
import matplotlib.pyplot as plt
import nibabel as nib
task_name="Task378_Parse2022"
task_folder_name="C:\\Users\\Administrateur\\Desktop\\mmchalleneges2022\\Parse2021\\NNUnet_data_prepartion"
train_label_dir=os.path.join(task_folder_name,"labelsTr")
test_dir=os.path.join(task_folder_name,"imagesTs")
for j in os.listdir(test_dir):
    print(j)
#for colab users only - keep the base directory same as above
import os
overwrite_json_file = True #make it True if you want to overwrite the dataset.json file in Task_folder
json_file_exist = False

if os.path.exists(os.path.join(task_folder_name,'dataset.json')):
    print('dataset.json already exist!')
    json_file_exist = True

if json_file_exist==False or overwrite_json_file:

    json_dict = OrderedDict()
    json_dict['name'] = task_name
    json_dict['description'] = "Pulmonary Artery Segmentation Challenge 2022 "
    json_dict['tensorImageSize'] = "3D"
    json_dict['reference'] = "see challenge website"
    json_dict['licence'] = "see challenge website"
    json_dict['release'] = "0.0"

    #you may mention more than one modality
    json_dict['modality'] = {
        "0": "CT"
    }
    #labels+1 should be mentioned for all the labels in the dataset
    json_dict['labels'] = {
        "0": "background",
        "1": "Pulmonary ARtery ",
    }    
    train_ids = os.listdir(train_label_dir)
    #test_ids = os.listdir(test_dir)
    json_dict['numTraining'] = len(train_ids)
    json_dict['numTest'] = []

    #no modality in train image and labels in dataset.json 
    json_dict['training'] = [{'image': "./imagesTr/%s" % i.split('_')[0], "label": "./labelsTr/%s" % i} for i in train_ids]

    #removing the modality from test image name to be saved in dataset.json
    json_dict['test'] = []
    with open(os.path.join(task_folder_name,"dataset.json"), 'w') as f:
        json.dump(json_dict, f, indent=4, sort_keys=True)

    if os.path.exists(os.path.join(task_folder_name,'dataset.json')):
        if json_file_exist==False:
            print('dataset.json created!')
        else: 
            print('dataset.json overwritten!')
#%% create json for instance dataset
import shutil
from collections import OrderedDict
import json
import numpy as np
import os

#visualization of the dataset
import matplotlib.pyplot as plt
import nibabel as nib
task_name="Task175_Parse2022"
task_folder_name="C:\\Users\\Administrateur\\Desktop\\mmchalleneges2022\\Parse2021\\NNUnet_data_prepartion"
train_label_dir=os.path.join(task_folder_name,"labelsTr")
test_dir=os.path.join(task_folder_name,"imagesTs")
for j in os.listdir(test_dir):
    print(j)
#for colab users only - keep the base directory same as above
import os
overwrite_json_file = True #make it True if you want to overwrite the dataset.json file in Task_folder
json_file_exist = False

if os.path.exists(os.path.join(task_folder_name,'dataset.json')):
    print('dataset.json already exist!')
    json_file_exist = True

if json_file_exist==False or overwrite_json_file:

    json_dict = OrderedDict()
    json_dict['name'] = task_name
    json_dict['description'] = "Pulmonary Artery Segmentation Challenge 2022 "
    json_dict['tensorImageSize'] = "3D"
    json_dict['reference'] = "see challenge website"
    json_dict['licence'] = "see challenge website"
    json_dict['release'] = "0.0"

    #you may mention more than one modality
    json_dict['modality'] = {
        "0": "CT"
    }
    #labels+1 should be mentioned for all the labels in the dataset
    json_dict['labels'] = {
        "0": "background",
        "1": "Pulmonary ARtery ",
    }    
    train_ids = os.listdir(train_label_dir)
    test_ids = os.listdir(test_dir)
    json_dict['numTraining'] = len(train_ids)
    json_dict['numTest'] = len(test_ids)

    #no modality in train image and labels in dataset.json 
    json_dict['training'] = [{'image': "./imagesTr/%s" % i.split('_')[0], "label": "./labelsTr/%s" % i} for i in train_ids]

    #removing the modality from test image name to be saved in dataset.json
    json_dict['test'] = ["./imagesTs/%s" % i.split('_')[0]+'.nii.gz' for i in test_ids]

    with open(os.path.join(task_folder_name,"dataset.json"), 'w') as f:
        json.dump(json_dict, f, indent=4, sort_keys=True)

    if os.path.exists(os.path.join(task_folder_name,'dataset.json')):
        if json_file_exist==False:
            print('dataset.json created!')
        else: 
            print('dataset.json overwritten!')
            
#%%

#%%
import os
import nibabel as nib
import glob
import SimpleITK as sitk
path='C:\\Users\\Administrateur\\Desktop\\mmchalleneges2022\\Parse2021\\train\\train'
#pathim=os.path.join(path,'image')
#pathlabel=os.path.join(path,'label')
lstdir=sorted(os.listdir(path))
#### get images path
#patients_pattern_img = os.path.join(path,'*','*image')
patients_images = glob.glob(os.path.join(path,'*','*image','*'))
###### get label paths
#img=sorted(glob.glob(os.path.join(lstdir,'*.nii.gz')))
#label=sorted(glob.glob(os.path.join(pathlabel,'*.nii.gz')))
#patients_pattern_label = os.path.join(path,'*','*label')
patients_label = glob.glob(os.path.join(path,'*','*label','*'))

#n_objct=nib.load(img[0]).get_data()
for i in range(0,len(patients_images)):
    n_objct=nib.load(patients_images[i]).get_data()
    label1=nib.load(patients_label[i]).get_data()
    #print('image:',n_objct.shape)
    #print('label:',label1.shape)
    imge=sitk.ReadImage(patients_images[i])
    print(imge.GetSpacing())
    break