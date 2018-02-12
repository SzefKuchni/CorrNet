__author__ = 'Sarath'

import numpy as np
import os
import sys
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from matplotlib import interactive
from plot_utils import combine_views


sys.path.append("../Model/")
from corrnet_4 import CorrNet  
#from reconstruct_corrnet3 import mse_dis  why cannot??
ID=200
SHAPE=[14,14]
def plot_part(real,part,title,n,MODE_NUM):
    
    #print len(part)
    part.insert(n,real)
    reshaped_parts=[]
    for ii in range(MODE_NUM):
        new =part[ii][ID,:]
        reshaped_parts.append(new.reshape(SHAPE[0],SHAPE[1]))
        
    res=combine_views(reshaped_parts,MODE_NUM)
    #print res.shape

    plt.subplot(2,2,n+1)
    plt.imshow(res,cmap='gray')
    plt.title(title)

def create_folder(folder):

	if not os.path.exists(folder):
		os.makedirs(folder)
def mse_dis(mat1,mat2):
    #dis= np.mean(np.sqrt(np.sum((mat1-mat2)**2,axis=1)))
    err = (mat1-mat2)**2
    dis = np.mean(err)
    return dis

def reconstruct_error(model, mat1,mat2):
    r1=model.reconstruct(mat1,1)
    r2=model.reconstruct(mat2,2)
    err12=mse_dis(mat2,r1[1].eval())
    err21=mse_dis(mat1,r2[0].eval())

    return err12,err21,r1[1].eval(),r2[0].eval()
    
    
def reconstruct_folder(src_folder,model, folder_name,v1_name,v2_name):
    mat1 = np.load(src_folder+folder_name+"/view"+v1_name+".npy")
    mat2 = np.load(src_folder+folder_name+"/view"+v2_name+".npy")
     
    err12,err21,new12,new21=reconstruct_error(model,mat1,mat2)

    return err12,err21,mat1,mat2,new12,new21


def get_reconstruct_view(src_folder, tgt_folder, sub_folder_name,v1_name,v2_name,mode_num):
    model = CorrNet()

    plist = pickle.load(open(tgt_folder+"params.pck","rb"))
    
    
    model.init(l_rate=plist["l_rate"],
              tied=plist["tied"],
              n_visible=plist["n_visible"], 
              n_hidden=plist["n_hidden"], lamda=plist["lamda"],  
              hidden_activation=plist["hidden_activation"], output_activation=plist["output_activation"],
               op_folder=tgt_folder,MODE_NUM=mode_num)
    
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess,tgt_folder)
        
#        detfile=open("res/ABC_direct_reconstruction.txt","a")
#        detfile.write("/n")
        err12,err21,mat1,mat2,new12,new21 =reconstruct_folder(src_folder,model,sub_folder_name,v1_name,v2_name)
#        detfile.write(tgt_folder+sub_folder_name+"/t")
#        detfile.write(str(err1_2) + "\t"+str(err2_1) + "\n")
#    
    tf.reset_default_graph()
    return err12,err21,mat1,mat2,new12,new21

def train_valid_test(subfolder_name):

    
    src_name="input/M_ABCD"
    tgt_name="output/ABCD/T_"
    MODE_NUM=4
    
    errs = np.zeros((MODE_NUM,MODE_NUM),dtype=np.float32)
    parts={}
    real={}
    for ii in range(MODE_NUM):
        parts[str(ii+1)]=[]
    
    for ii in range(MODE_NUM):
        view_name=str(ii+1)
        
        for jj in range(ii+1,MODE_NUM):
            v1_name=str(ii+1)
            v2_name=str(jj+1)
            src_folder=src_name+"/matpic/"
            tgt_folder=tgt_name+v1_name+v2_name+"/"
            err12,err21,mat1,mat2,new12,new21=get_reconstruct_view(src_folder,tgt_folder,subfolder_name,v1_name,v2_name,2)
            errs[ii,jj]= err12
            errs[jj,ii]=err21
            parts[view_name].append(new12)
            parts[str(jj+1)].append(new21)
            real[str(ii+1)]= mat1
            real[str(jj+1)]= mat2
    err_sum = np.sum(errs,axis = 1)/(MODE_NUM-1)
    detfile=open("res/direct_train.txt","a")
    
    detfile.write("\n"+subfolder_name+"\t")
    print ("\n"+subfolder_name+"\t")
    for ii in range(MODE_NUM):
        detfile.write(str(err_sum[ii])+"\t")
        print (str(err_sum[ii])+"\t")
    detfile.close() 
    
    for ii in range (MODE_NUM):
    #    view_name=str(ii+1)
    #    true_view=real[view_name]
    #    parts[view_name].insert(ii,true_view)
        
        plot_part(real[str(ii+1)],parts[str(ii+1)],"Recover from view"+str(ii+1),ii,MODE_NUM)
    
    plt.show()



if __name__=="__main__":
    
    train_valid_test("train")
    train_valid_test("valid")
    train_valid_test("test")



    

