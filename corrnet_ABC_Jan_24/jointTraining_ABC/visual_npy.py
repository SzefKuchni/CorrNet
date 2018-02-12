# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 18:14:19 2017

@author: zhangyaqian
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

#Name="TGT_DIR_LR/project/test-view1.npy"
#Name2="TGT_DIR_LR/project/test-view1.npy"

id="2"
num = 5
SHAPE=9#14#9

#y=np.load(Name)[num,:].reshape(28,SHAPE)
#plt.imshow(y)
#plt.show()
#y=np.load(Name2)[num,:].reshape(28,SHAPE)
#plt.imshow(y)
#plt.show()

folder="input/M_BC/matpic1/train/"
print np.load(folder+id+"_left.npy").shape




left = np.load(folder+id+"_left.npy")[num,:].reshape(28,SHAPE)
plt.imshow(left)#,cmap='gray')
plt.show()

#if SHAPE == 9:
#    mid = np.load(folder+id+"_middle.npy")[num,:].reshape(28,SHAPE)
#    plt.imshow(mid)#,cmap='gray')
#    plt.show()
    
right = np.load(folder+id+"_right.npy")[num,:].reshape(28,SHAPE)

plt.imshow(right)#,cmap='gray')
plt.show()

#full=np.concatenate((left,mid,right), axis=1)
#plt.imshow(full, cmap='gray')

#
#name="TGT_DIR_UPLR_less/project/test-view1.npy"
#pic=np.load(name)
#print pic.shape
##view= np.load(folder)

#plt.imshow(pic)#,cmap='gray')
#plt.show()