__author__ = 'Yaqian'

import numpy as np
import os
import sys
from utils import visualize

def create_folder(folder):

    if not os.path.exists(folder):
        os.makedirs(folder)


def get_mat(fname):

    file = open(fname,"r")
    mat = list()
    for line in file:
        line = line.strip().split()
        mat.append(line)
    mat = np.asarray(mat,dtype="float32")
    return mat


def get_mat1(folder,mats,mode_num):
    file = open(folder+"ip.txt","w")
    flag = 0
    smats=[]
    for m in range(mode_num):
        smat=np.zeros((1000,len(mats[m][0])))
        smats.append(smat)

    i = 0
    length=len(mats[0])
    while(i!=length):
        flag = 1
        for m in range(mode_num):
            smats[m][i%1000] = mats[m][i]

        i+=1
        if(i%1000==0):
            for m in range(mode_num):
                view_name=str(m+1)
                np.save(folder+str(i/1000)+"_"+view_name,smats[m])
            file.write("xy,dense,"+folder+str(i/1000)+",1000\n")

            for m in range(mode_num):
                 smats[m]=np.zeros((1000,len(mats[m][0])))
            flag = 0

    if(flag!=0):
         for m in range(mode_num):
              view_name=str(m+1)
              np.save(folder+str((i/1000)+1)+"_"+view_name,smats[m])
         file.write("xy,dense,"+folder+str((i/1000) +1)+","+str(i%1000)+"\n")
    file.close()

def build_matpic(folder,sub_folder_name,sub_output_name,mode_num):
    for ii in range(mode_num):
        view_name=str(ii+1)
        mat = get_mat(folder+sub_folder_name+"-view"+view_name+".txt")
        np.save(folder+"matpic/"+sub_output_name+"/view"+view_name,mat)
    np.save(folder+"matpic/"+sub_output_name+"/labels",get_mat(folder+sub_folder_name+"-labels.txt"))


def converter(folder,mode_num):


    create_folder(folder+"matpic1/")
    create_folder(folder+"matpic1/train")
    create_folder(folder+"matpic1/valid")
    create_folder(folder+"matpic1/test")
    mats = []
    for ii in range(mode_num):
        view_name=str(ii+1)
        mat = get_mat(folder+"train-view"+view_name+".txt")
        mats.append(mat)
    get_mat1(folder+"matpic1/train/",mats,mode_num)


    create_folder(folder+"matpic/")
    create_folder(folder+"matpic/train")
    create_folder(folder+"matpic/valid")
    create_folder(folder+"matpic/test")
    build_matpic(folder,"valid1","train",mode_num)
    build_matpic(folder,"valid2","valid",mode_num)
    build_matpic(folder,"test","test",mode_num)

    visualize(folder,mode_num)



if __name__ == '__main__':
    converter(sys.argv[1],int(sys.argv[2])) # e.g. python create_data4.py input/M_ABC 3
