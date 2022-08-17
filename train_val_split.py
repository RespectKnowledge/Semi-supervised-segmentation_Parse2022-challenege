# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 08:18:32 2022

@author: Administrateur
"""

import os
import pandas as pd
import nibabel as nib
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
patht=pd.read_csv('/home/imranr/monabdul/Parse2022/train_fold0_parse.csv')
path_save='/home/imranr/monabdul/Parse2022/train_val_split/train'
patht['PatientID'][0]
datapath='/home/imranr/monabdul/Parse2022/'
for i in range(0,len(patht)):
    pathf=patht['PatientID'][i]
    pattren=pathf.split('/')[-4]+'/'+pathf.split('/')[-3]+'/'+pathf.split('/')[-2]
    labelp=pattren.replace('image','label')
    p=os.path.join(os.path.join(path_save,pattren))
    createFolder(p)
    p1=os.path.join(os.path.join(path_save,labelp))
    createFolder(p1)
    img=nib.load(os.path.join(datapath,pathf))
    mask=nib.load(os.path.join(datapath,pathf.replace('image','label')))
    #break
    nib.save(img,os.path.join(p,pathf.split('/')[-1]))
    nib.save(mask,os.path.join(p1,pathf.split('/')[-1]))
    print(pathf.split('/')[-1])
print('done')
    #break
    
################ validation
# import os
# import pandas as pd
# import nibabel as nib
# def createFolder(directory):
#     try:
#         if not os.path.exists(directory):
#             os.makedirs(directory)
#     except OSError:
#         print ('Error: Creating directory. ' +  directory)
# patht=pd.read_csv('/home/imranr/monabdul/Parse2022/valid_fold0_parse.csv')
# path_save='/home/imranr/monabdul/Parse2022/train_val_split/valid'
# patht['PatientID'][0]
# datapath='/home/imranr/monabdul/Parse2022/'
# for i in range(0,len(patht)):
#     pathf=patht['PatientID'][i]
#     pattren=pathf.split('/')[-4]+'/'+pathf.split('/')[-3]+'/'+pathf.split('/')[-2]
#     labelp=pattren.replace('image','label')
#     p=os.path.join(os.path.join(path_save,pattren))
#     createFolder(p)
#     p1=os.path.join(os.path.join(path_save,labelp))
#     createFolder(p1)
#     img=nib.load(os.path.join(datapath,pathf))
#     mask=nib.load(os.path.join(datapath,pathf.replace('image','label')))
#     #break
#     nib.save(img,os.path.join(p,pathf.split('/')[-1]))
#     nib.save(mask,os.path.join(p1,pathf.split('/')[-1]))
#     print(pathf.split('/')[-1])
# print('done')