import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
import sys

def get_mat(fname):

	file = open(fname,"r")
	mat = list()
	for line in file:
		line = line.strip().split()
		mat.append(line)
	mat = np.asarray(mat,dtype="float32")
	return mat

def split_image(matrix):
    size = 9
    return matrix[:,:,:size], matrix[:,:,size:(size*2)], matrix[:,:,(size*2):(size*3)]

def converter(folder):
    mat1 = get_mat(folder+"/test-view1.txt")
    mat2 = get_mat(folder+"/test-view2.txt")
    full=np.concatenate((mat1.reshape(mat1.shape[0],28,14), mat2.reshape(mat2.shape[0],28,14)), axis=2)
    A, B, C = split_image(full)
    A_save=A.reshape(A.shape[0],252)
    np.savetxt(folder+"/test-view1.txt", A_save.round(4), fmt="%s")
    B_save=B.reshape(B.shape[0],252)
    np.savetxt(folder+"/test-view2.txt", B_save.round(4), fmt="%s")
    C_save=C.reshape(C.shape[0],252)
    np.savetxt(folder+"/test-view3.txt", C_save.round(4), fmt="%s")
    
    mat1 = get_mat(folder+"/train-view1.txt")
    mat2 = get_mat(folder+"/train-view2.txt")
    full=np.concatenate((mat1.reshape(mat1.shape[0],28,14), mat2.reshape(mat2.shape[0],28,14)), axis=2)
    A, B, C = split_image(full)
    A_save=A.reshape(A.shape[0],252)
    np.savetxt(folder+"/train-view1.txt", A_save.round(4), fmt="%s")
    B_save=B.reshape(B.shape[0],252)
    np.savetxt(folder+"/train-view2.txt", B_save.round(4), fmt="%s")
    C_save=C.reshape(C.shape[0],252)
    np.savetxt(folder+"/train-view3.txt", C_save.round(4), fmt="%s")
    
    mat1 = get_mat(folder+"/valid1-view1.txt")
    mat2 = get_mat(folder+"/valid1-view2.txt")
    full=np.concatenate((mat1.reshape(mat1.shape[0],28,14), mat2.reshape(mat2.shape[0],28,14)), axis=2)
    A, B, C = split_image(full)
    A_save=A.reshape(A.shape[0],252)
    np.savetxt(folder+"/valid1-view1.txt", A_save.round(4), fmt="%s")
    B_save=B.reshape(B.shape[0],252)
    np.savetxt(folder+"/valid1-view2.txt", B_save.round(4), fmt="%s")
    C_save=C.reshape(C.shape[0],252)
    np.savetxt(folder+"/valid1-view3.txt", C_save.round(4), fmt="%s")
    
    mat1 = get_mat(folder+"/valid2-view1.txt")
    mat2 = get_mat(folder+"/valid2-view2.txt")
    full=np.concatenate((mat1.reshape(mat1.shape[0],28,14), mat2.reshape(mat2.shape[0],28,14)), axis=2)
    A, B, C = split_image(full)
    A_save=A.reshape(A.shape[0],252)
    np.savetxt(folder+"/valid2-view1.txt", A_save.round(4), fmt="%s")
    B_save=B.reshape(B.shape[0],252)
    np.savetxt(folder+"/valid2-view2.txt", B_save.round(4), fmt="%s")
    C_save=C.reshape(C.shape[0],252)
    np.savetxt(folder+"/valid2-view3.txt", C_save.round(4), fmt="%s")

converter(sys.argv[1])

