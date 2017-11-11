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
    half = 14
    return matrix[:,:,:half], matrix[:,:,half:]

def converter(folder):
    mat1 = get_mat(folder+"/test-view1.txt")
    mat2 = get_mat(folder+"/test-view2.txt")
    full=np.concatenate((mat1.reshape(mat1.shape[0],28,14), mat2.reshape(mat2.shape[0],28,14)), axis=2)
    full_rotated=rotate(full, 90, axes=(1,2))
    up, down = split_image(full_rotated)
    up_save=up.reshape(up.shape[0],392)
    np.savetxt(folder+"/test-view1.txt", up_save.round(4), fmt="%s")
    down_save=down.reshape(down.shape[0],392)
    np.savetxt(folder+"/test-view2.txt", down_save.round(4), fmt="%s")
    
    mat1 = get_mat(folder+"/train-view1.txt")
    mat2 = get_mat(folder+"/train-view2.txt")
    full=np.concatenate((mat1.reshape(mat1.shape[0],28,14), mat2.reshape(mat2.shape[0],28,14)), axis=2)
    full_rotated=rotate(full, 90, axes=(1,2))
    up, down = split_image(full_rotated)
    up_save=up.reshape(up.shape[0],392)
    np.savetxt(folder+"/train-view1.txt", up_save.round(4), fmt="%s")
    down_save=down.reshape(down.shape[0],392)
    np.savetxt(folder+"/train-view2.txt", down_save.round(4), fmt="%s")
    
    mat1 = get_mat(folder+"/valid1-view1.txt")
    mat2 = get_mat(folder+"/valid1-view2.txt")
    full=np.concatenate((mat1.reshape(mat1.shape[0],28,14), mat2.reshape(mat2.shape[0],28,14)), axis=2)
    full_rotated=rotate(full, 90, axes=(1,2))
    up, down = split_image(full_rotated)
    up_save=up.reshape(up.shape[0],392)
    np.savetxt(folder+"/valid1-view1.txt", up_save.round(4), fmt="%s")
    down_save=down.reshape(down.shape[0],392)
    np.savetxt(folder+"/valid1-view2.txt", down_save.round(4), fmt="%s")
    
    mat1 = get_mat(folder+"/valid2-view1.txt")
    mat2 = get_mat(folder+"/valid2-view2.txt")
    full=np.concatenate((mat1.reshape(mat1.shape[0],28,14), mat2.reshape(mat2.shape[0],28,14)), axis=2)
    full_rotated=rotate(full, 90, axes=(1,2))
    up, down = split_image(full_rotated)
    up_save=up.reshape(up.shape[0],392)
    np.savetxt(folder+"/valid2-view1.txt", up_save.round(4), fmt="%s")
    down_save=down.reshape(down.shape[0],392)
    np.savetxt(folder+"/valid2-view2.txt", down_save.round(4), fmt="%s")

converter(sys.argv[1])

