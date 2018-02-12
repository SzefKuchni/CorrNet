__author__ = 'Sarath'

import numpy
import os
import sys
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from matplotlib import interactive
from utils import combine_views,get_view_name,get_setting_from_name_code

sys.path.append("../Model/")
from corrnet_4 import CorrNet


ID = 200
SHAPE_ORG=[28,28]
DIR=[2,2] 


def create_folder(folder):

	if not os.path.exists(folder):
		os.makedirs(folder)
def plot_part(output,real,n,title,MODE_NUM,SHAPE):
    parts=[]
    if n == 1:
        new=real[0]
    else:
        #new=real[0]
        new=output[0].eval()
    new = new[ID,:].reshape(SHAPE[0],SHAPE[1])
    parts.append(new)
    for ii in range(1,MODE_NUM):
        if (ii +1) ==  n:
            new = real[ii]
        else:
            #new = real[ii]
            new = output[ii].eval()
        new= new[ID,:].reshape(SHAPE[0],SHAPE[1])
        parts.append(new)
        
    res = combine_views(parts,MODE_NUM)
    plt.subplot(DIR[0],DIR[1],n)
    plt.imshow(res,cmap='gray')
    plt.title(title)

    
def mse_dis(mat1,mat2):
    #dis= numpy.mean(numpy.sqrt(numpy.sum((mat1-mat2)**2,axis=1)))
    err = (mat1-mat2)**2
    dis = numpy.mean(err)
    return dis

def reconstruct_folder(model, folder_name,src_folder,tgt_folder,MODE_NUM,v1_name,v2_name,SHAPE):
    print (folder_name+" reconstruction error")
    detfile=open("res/ABC_joint_reconstruction.txt","a")
    detfile.write("\n"+tgt_folder+folder_name+"\t")
    
    real=[]
    err_list=[]
    for ii in range(MODE_NUM):
        view_name=get_view_name(ii,MODE_NUM,v1_name,v2_name)
        mat = numpy.load(src_folder+folder_name+"/view"+view_name+".npy")
        real.append(mat)
    for ii in range(MODE_NUM):
        
        output = model.reconstruct(real[ii],ii+1)
        #plot_part(output,real,ii+1,"recoved from view"+str(ii+1),MODE_NUM,SHAPE)
        
        err_sum=0
        for jj in range(MODE_NUM):
            
            if(jj!=ii):
                err = mse_dis(output[jj].eval(),real[jj])
                print "view"+str(ii+1)+" to "+str(jj+1)+"recover error: ",str(err)
                err_sum += err
        err_sum = err_sum/(MODE_NUM-1)       
        err_list.append(err_sum) # err_sum is the cross reconstruction err for view ii
        print err_sum
        detfile.write(str(err_sum)+"\t")
    #plt.show()

    detfile.close()
                
    return err_list
    
def get_reconstruct_view(src_folder,tgt_folder,sub_folder_name,MODE_NUM,v1_name,v2_name,SHAPE):

    model = CorrNet()
    
    folder=tgt_folder
    plist = pickle.load(open(folder+"params.pck","rb"))
    
    model.init(l_rate=plist["l_rate"],tied=plist["tied"], n_hidden=plist["n_hidden"], lamda=plist["lamda"],n_visible=plist["n_visible"],
              hidden_activation=plist["hidden_activation"], output_activation=plist["output_activation"],
               op_folder=folder,MODE_NUM = MODE_NUM)
    
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess,tgt_folder)
        reconstruct_folder(model,sub_folder_name,src_folder,tgt_folder,MODE_NUM,v1_name,v2_name,SHAPE)
    tf.reset_default_graph()
def start_reconstruct(src,tgt, SHAPE):
    size = SHAPE[0]*SHAPE[1]#14*14 
    src=  src+"matpic/"
    mode_num,v1_name,v2_name = get_setting_from_name_code(tgt)
    get_reconstruct_view(src,tgt,"train",mode_num,v1_name,v2_name,SHAPE)
    get_reconstruct_view(src,tgt,"test",mode_num,v1_name,v2_name,SHAPE)
    get_reconstruct_view(src,tgt,"valid",mode_num,v1_name,v2_name,SHAPE)

if __name__=='__main__':
    
        
    src=  sys.argv[1]
    tgt=  sys.argv[2]
    SHAPE=[14,14]
    start_reconstruct(src,tgt,SHAPE) #e.g. python reconstruct_corrnet4.py input/M_ABCD/ output/T_ABCD/





