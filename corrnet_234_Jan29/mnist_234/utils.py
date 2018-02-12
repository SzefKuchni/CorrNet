# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 11:57:49 2018

@author: zhangyaqian
"""
import numpy as np

import matplotlib.pyplot as plt

def combine_views(parts,MODE_NUM):
    if MODE_NUM==4:
        res=np.concatenate((np.concatenate((parts[0],parts[1]),axis=1),np.concatenate((parts[2],parts[3]),axis=1)),axis=0)
    elif MODE_NUM == 3:
        res=np.concatenate((parts[0],parts[1],parts[2]),axis=1)
    else:
        res=np.concatenate((parts[0],parts[1]),axis=1)
        
    return res
    
def get_view_name(i,mode_num,v1,v2):
    
    if(mode_num == 2): ## for 2 modes, need to specify the view No.
        if(i==0):
            return v1
        else:
            return v2
    else:    ## for 3 or 4 view, just follow the name in the folder   
        return str(i+1)
def get_setting_from_name_code(tgt_folder):
    
    #output/ABCD/T_
    for i in range(len(tgt_folder)):
        if(tgt_folder[i]=="_"):
            break
    start = i+1
    mode=tgt_folder[start:]
    mode_num= len(mode)-1#int(sys.argv[3])
    view1_name=mode[0]
    view2_name=mode[1]
    print "views :",mode, "view num: ",mode_num
 

    return mode_num, view1_name,view2_name
    
    
def visualize(folder, mode_num):
    id="1"
    num = 25
    folder=folder+"matpic1/train/"
    if(mode_num == 4):
        SHAPE=[14,14]
    elif(mode_num == 3):
        SHAPE=[28,9]
    elif(mode_num == 2):
        SHAPE = [28,14]
    else:
        print ("error with mode num"+ str(mode_num))
    for i in range(mode_num):
        view_name=str(i+1)
        A= np.load(folder+id+"_"+view_name+".npy")[num,:].reshape(SHAPE[0],SHAPE[1])
        plt.imshow(A,cmap='gray')
        plt.show()
    
