import numpy as np
#import matplotlib.pyplot as plt
#from scipy.ndimage import rotate
import sys
from utils import visualize

def get_mat(fname):

	file = open(fname,"r")
	mat = list()
	for line in file:
		line = line.strip().split()
		mat.append(line)
	mat = np.asarray(mat,dtype="float32")
	return mat

def split_in_three(matrix):
    size = 9
    return matrix[:,:,:size], matrix[:,:,size:(size*2)], matrix[:,:,(size*2):(size*3)]
def split_in_four(matrix):
    size = 14
    return matrix[:,:size,:size], matrix[:,:size,size:(size*2)], matrix[:,size:size*2,:size],matrix[:,size:size*2,size:size*2]

def convert_folder(folder,sub_folder_name,mode_num):
    
    mat1 = get_mat(folder+"/"+sub_folder_name+"-view1.txt")
    mat2 = get_mat(folder+"/"+sub_folder_name+"-view2.txt")
    full=np.concatenate((mat1.reshape(mat1.shape[0],28,14), mat2.reshape(mat2.shape[0],28,14)), axis=2)
    if( mode_num == 4):
        A, B, C ,D= split_in_four(full)
        DIM=14*14 #252
        A_save=A.reshape(A.shape[0],DIM)
        np.savetxt(folder+"/"+sub_folder_name+"-view1.txt", A_save.round(4), fmt="%s")
        B_save=B.reshape(B.shape[0],DIM)
        np.savetxt(folder+"/"+sub_folder_name+"-view2.txt", B_save.round(4), fmt="%s")
        C_save=C.reshape(C.shape[0],DIM)
        np.savetxt(folder+"/"+sub_folder_name+"-view3.txt", C_save.round(4), fmt="%s")
        D_save=D.reshape(D.shape[0],DIM)
        np.savetxt(folder+"/"+sub_folder_name+"-view4.txt", D_save.round(4), fmt="%s")
    elif (mode_num == 3):
        A, B, C = split_in_three(full)
        DIM=252
        A_save=A.reshape(A.shape[0],DIM)
        np.savetxt(folder+"/"+sub_folder_name+"-view1.txt", A_save.round(4), fmt="%s")
        B_save=B.reshape(B.shape[0],DIM)
        np.savetxt(folder+"/"+sub_folder_name+"-view2.txt", B_save.round(4), fmt="%s")
        C_save=C.reshape(C.shape[0],DIM)
        np.savetxt(folder+"/"+sub_folder_name+"-view3.txt", C_save.round(4), fmt="%s")
    else:
        print ("there is a problem with mode_num",str(mode_num))
    
def converter(folder,mode_num):
    print ("converting data in "+folder+" into "+mode_num+ " parts")
    mode_num=int(mode_num)
    convert_folder(folder,"test",mode_num)
    convert_folder(folder,"train",mode_num)
    convert_folder(folder,"valid1",mode_num)
    convert_folder(folder,"valid2",mode_num)

    
#    mat1 = get_mat(folder+"/test-view1.txt")
#    mat2 = get_mat(folder+"/test-view2.txt")
#    full=np.concatenate((mat1.reshape(mat1.shape[0],28,14), mat2.reshape(mat2.shape[0],28,14)), axis=2)
#    A, B, C = split_image(full)
#    A_save=A.reshape(A.shape[0],252)
#    np.savetxt(folder+"/test-view1.txt", A_save.round(4), fmt="%s")
#    B_save=B.reshape(B.shape[0],252)
#    np.savetxt(folder+"/test-view2.txt", B_save.round(4), fmt="%s")
#    C_save=C.reshape(C.shape[0],252)
#    np.savetxt(folder+"/test-view3.txt", C_save.round(4), fmt="%s")
#    
#    mat1 = get_mat(folder+"/train-view1.txt")
#    mat2 = get_mat(folder+"/train-view2.txt")
#    full=np.concatenate((mat1.reshape(mat1.shape[0],28,14), mat2.reshape(mat2.shape[0],28,14)), axis=2)
#    A, B, C = split_image(full)
#    A_save=A.reshape(A.shape[0],252)
#    np.savetxt(folder+"/train-view1.txt", A_save.round(4), fmt="%s")
#    B_save=B.reshape(B.shape[0],252)
#    np.savetxt(folder+"/train-view2.txt", B_save.round(4), fmt="%s")
#    C_save=C.reshape(C.shape[0],252)
#    np.savetxt(folder+"/train-view3.txt", C_save.round(4), fmt="%s")
#    
#    mat1 = get_mat(folder+"/valid1-view1.txt")
#    mat2 = get_mat(folder+"/valid1-view2.txt")
#    full=np.concatenate((mat1.reshape(mat1.shape[0],28,14), mat2.reshape(mat2.shape[0],28,14)), axis=2)
#    A, B, C = split_image(full)
#    A_save=A.reshape(A.shape[0],252)
#    np.savetxt(folder+"/valid1-view1.txt", A_save.round(4), fmt="%s")
#    B_save=B.reshape(B.shape[0],252)
#    np.savetxt(folder+"/valid1-view2.txt", B_save.round(4), fmt="%s")
#    C_save=C.reshape(C.shape[0],252)
#    np.savetxt(folder+"/valid1-view3.txt", C_save.round(4), fmt="%s")
#    
#    mat1 = get_mat(folder+"/valid2-view1.txt")
#    mat2 = get_mat(folder+"/valid2-view2.txt")
#    full=np.concatenate((mat1.reshape(mat1.shape[0],28,14), mat2.reshape(mat2.shape[0],28,14)), axis=2)
#    A, B, C = split_image(full)
#    A_save=A.reshape(A.shape[0],252)
#    np.savetxt(folder+"/valid2-view1.txt", A_save.round(4), fmt="%s")
#    B_save=B.reshape(B.shape[0],252)
#    np.savetxt(folder+"/valid2-view2.txt", B_save.round(4), fmt="%s")
#    C_save=C.reshape(C.shape[0],252)
#    np.savetxt(folder+"/valid2-view3.txt", C_save.round(4), fmt="%s")

converter(sys.argv[1], sys.argv[2])

