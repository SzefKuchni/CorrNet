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

left=get_mat("test-view1.txt")
right=get_mat("test-view2.txt")

plt.imshow(left[0].reshape(28,14), cmap='gray')
plt.imshow(right[0].reshape(28,14), cmap='gray')
plt.imshow(np.concatenate((left[0,].reshape(28,14), right[0,].reshape(28,14)), axis=1), cmap='gray')


#joining pictures

full=np.concatenate((left.reshape(left.shape[0],28,14), right.reshape(left.shape[0],28,14)), axis=2)

#rotating single image
test0=full[0]
plt.imshow(test0, cmap='gray')
test=np.rot90(test0)
plt.imshow(test, cmap='gray')

#rotating multiple images
full_rotated=rotate(full, 90, axes=(1,2))

plt.imshow(full_rotated[0], cmap='gray')
plt.imshow(full_rotated[1], cmap='gray')
plt.imshow(full_rotated[2], cmap='gray')

#slicing it into 2 halfs
test=full_rotated[0]
def split_image(matrix):
    half = 14
    return matrix[:,:half], matrix[:,half:]
up, down = split_image(test)

plt.imshow(up, cmap='gray')
plt.imshow(down, cmap='gray')

#slicing it into 2 halfs
def split_image(matrix):
    half = 14
    return matrix[:,:,:half], matrix[:,:,half:]
up, down = split_image(full_rotated)

plt.imshow(up[0], cmap='gray')
plt.imshow(up[1], cmap='gray')
plt.imshow(up[2], cmap='gray')
plt.imshow(down[0], cmap='gray')
plt.imshow(down[1], cmap='gray')
plt.imshow(down[2], cmap='gray')


#checkoing the results
left=get_mat("test-view1_test.txt")
right=get_mat("test-view2_test.txt")

plt.imshow(left[0].reshape(28,14), cmap='gray')
plt.imshow(right[0].reshape(28,14), cmap='gray')
plt.imshow(np.concatenate((left[0,].reshape(28,14), right[0,].reshape(28,14)), axis=1), cmap='gray')
