import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate

def get_mat(fname):

	file = open(fname,"r")
	mat = list()
	for line in file:
		line = line.strip().split()
		mat.append(line)
	mat = np.asarray(mat,dtype="float32")
	return mat

left=get_mat("MNIST_DIR_original/test-view1.txt")
right=get_mat("MNIST_DIR_original/test-view2.txt")

plt.imshow(left[0].reshape(28,14), cmap='gray')
plt.imshow(right[0].reshape(28,14), cmap='gray')
plt.imshow(np.concatenate((left[0,].reshape(28,14), right[0,].reshape(28,14)), axis=1), cmap='gray')

#joining pictures

full=np.concatenate((left.reshape(left.shape[0],28,14), right.reshape(left.shape[0],28,14)), axis=2)
plt.imshow(full[0])
plt.imshow(full[1])
plt.imshow(full[2])
plt.imshow(full[3])
plt.imshow(full[4])
plt.imshow(full[5])

#there is a problem with the shape
full.shape

#slicing it into 3 parts
def split_image(matrix):
    size = 9
    return matrix[:,:,:size], matrix[:,:,size:(size*2)], matrix[:,:,(size*2):(size*3)]
A, B, C = split_image(full)

plt.imshow(A[0], cmap='gray')
plt.imshow(A[1], cmap='gray')
plt.imshow(A[2], cmap='gray')
plt.imshow(A[3], cmap='gray')
plt.imshow(A[4], cmap='gray')
plt.imshow(A[5], cmap='gray')
plt.imshow(A[6], cmap='gray')
plt.imshow(A[7], cmap='gray')
plt.imshow(A[8], cmap='gray')

plt.imshow(B[0], cmap='gray')
plt.imshow(C[0], cmap='gray')

A.sum()/B.sum()*100
B.sum()
C.sum()/B.sum()*100

#how to save
A_save=A.reshape(A.shape[0],252)
