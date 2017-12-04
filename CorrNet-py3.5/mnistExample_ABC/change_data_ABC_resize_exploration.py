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
full=np.concatenate((left.reshape(left.shape[0],28,14), right.reshape(left.shape[0],28,14)), axis=2)
#resizing a numpy aray image stroed as matrix
import scipy

##############WORKS
plt.imshow(full[1], cmap='gray')
test=scipy.misc.imresize(full[1],[40,40])
plt.imshow(test, cmap='gray')
##############WORKS

#start
img=full[1]
plt.imshow(img, cmap='gray')

test=img>0
width=test.max(axis=0)
height=test.max(axis=1)

height=np.where(height==True)
height_min=height[0].min()
height_max=height[0].max()

width=np.where(width==True)
width_min=width[0].min()
width_max=width[0].max()

global_max=max((height_max,width_max))+1
global_min=min((height_min,width_min))

plt.imshow(img, cmap='gray')
img_cropped=img[global_min:global_max,global_min:global_max]
plt.imshow(img_cropped, cmap='gray')
img_cropped_resized=scipy.misc.imresize(img_cropped,[28,28])
plt.imshow(img_cropped_resized, cmap='gray')

# reszizing a image in jpeg/png format
from PIL import Image
im=Image.open("51_flat_plan.jpg")
im.resize([300,300])
im=Image.open("test.png")
im
im.resize([500,500])

