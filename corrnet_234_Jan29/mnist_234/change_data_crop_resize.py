import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
import sys
import scipy

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

def crop_resize(matrix):
    # for loop to iterate through full data set
    full_transformed=np.ndarray([matrix.shape[0],28,28])

    for i in range(matrix.shape[0]):
        img=matrix[i]
        test=img>0
        width=test.max(axis=0)
        height=test.max(axis=1)
        #idicators if there is information in horizontal and vertical lines
        height=np.where(height==True)
        height_min=height[0].min()
        height_max=height[0].max()
        
        width=np.where(width==True)
        width_min=width[0].min()
        width_max=width[0].max()
        #idicators if there is information in horizontal and vertical lines

        bounding_square_size=max(width_max-width_min,height_max-height_min)+1
        
        img=scipy.misc.imresize(img,[28,28])
        img_cropped=img[height_min:height_min+bounding_square_size, width_min:width_min+bounding_square_size]
        img_cropped_resized=scipy.misc.imresize(img_cropped,[28,28])
        full_transformed[i]=img_cropped_resized/255.0
    return(full_transformed)

def converter(folder):
    mat1 = get_mat(folder+"/test-view1.txt")
    mat2 = get_mat(folder+"/test-view2.txt")
    full=np.concatenate((mat1.reshape(mat1.shape[0],28,14), mat2.reshape(mat2.shape[0],28,14)), axis=2)
    full_transformed=crop_resize(full)
    left, right = split_image(full_transformed)
    left_save=left.reshape(left.shape[0],392)
    np.savetxt(folder+"/test-view1.txt", left_save.round(4), fmt="%s")
    right_save=right.reshape(right.shape[0],392)
    np.savetxt(folder+"/test-view2.txt", right_save.round(4), fmt="%s")
    
    mat1 = get_mat(folder+"/train-view1.txt")
    mat2 = get_mat(folder+"/train-view2.txt")
    full=np.concatenate((mat1.reshape(mat1.shape[0],28,14), mat2.reshape(mat2.shape[0],28,14)), axis=2)
    full_transformed=crop_resize(full)
    left, right = split_image(full_transformed)
    left_save=left.reshape(left.shape[0],392)
    np.savetxt(folder+"/train-view1.txt", left_save.round(4), fmt="%s")
    right_save=right.reshape(right.shape[0],392)
    np.savetxt(folder+"/train-view2.txt", right_save.round(4), fmt="%s")
    
    mat1 = get_mat(folder+"/valid1-view1.txt")
    mat2 = get_mat(folder+"/valid1-view2.txt")
    full=np.concatenate((mat1.reshape(mat1.shape[0],28,14), mat2.reshape(mat2.shape[0],28,14)), axis=2)
    full_transformed=crop_resize(full)
    left, right = split_image(full_transformed)
    left_save=left.reshape(left.shape[0],392)
    np.savetxt(folder+"/valid1-view1.txt", left_save.round(4), fmt="%s")
    right_save=right.reshape(right.shape[0],392)
    np.savetxt(folder+"/valid1-view2.txt", right_save.round(4), fmt="%s")
    
    mat1 = get_mat(folder+"/valid2-view1.txt")
    mat2 = get_mat(folder+"/valid2-view2.txt")
    full=np.concatenate((mat1.reshape(mat1.shape[0],28,14), mat2.reshape(mat2.shape[0],28,14)), axis=2)
    full_transformed=crop_resize(full)
    left, right = split_image(full_transformed)
    left_save=left.reshape(left.shape[0],392)
    np.savetxt(folder+"/valid2-view1.txt", left_save.round(4), fmt="%s")
    right_save=right.reshape(right.shape[0],392)
    np.savetxt(folder+"/valid2-view2.txt", right_save.round(4), fmt="%s")

converter(sys.argv[1])

