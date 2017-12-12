import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
import cv2

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
img=full[4200]
plt.imshow(img, cmap='gray')

test=img>0
width=test.max(axis=0)
height=test.max(axis=1)
#idicators if there is information in horizontal and vertical lines
plt.imshow(img, cmap='gray')
width
height

height=np.where(height==True)
height_min=height[0].min()
height_max=height[0].max()
height_min
height_max

width=np.where(width==True)
width_min=width[0].min()
width_max=width[0].max()
#idicators if there is information in horizontal and vertical lines
plt.imshow(img, cmap='gray')
width_min
width_max

global_max=max((height_max,width_max))+1
global_min=min((height_min,width_min))

global_max
global_min

cut=max(min(27-height_max,width_min),0)
cut2=max(min(27-width_max,height_min),0)
cut
cut2

plt.imshow(img, cmap='gray')
img_cropped=img[cut2:global_max-cut,cut:global_max-cut2]
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

# for loop to iterate through full data set
full_transformed=np.ndarray([10000,28,28])

for i in range(0, 10):
    img=full[i]
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

    global_max=max((height_max,width_max))+1
    global_min=min((height_min,width_min))

    cut=max(min(27-height_max,width_min),0)
    cut2=max(min(27-width_max,height_min),0)
    
    img=scipy.misc.imresize(img,[28,28])
    img_cropped=img[cut2:global_max-cut,cut:global_max-cut2]
    img_cropped_resized=scipy.misc.imresize(img_cropped,[28,28])
    full_transformed[i]=img_cropped_resized
    #monitoring progress
    if i%1==0:
        print(i)
    #checking results
    if i<=10:
        im_original = Image.fromarray(img).convert("RGB")
        im_transformed = Image.fromarray(img_cropped_resized).convert("RGB")
        name_original="plots/image_"+str(i)+".png"
        name_transformed="plots/image_"+str(i)+"_transformed.png"
        im_original.save(name_original)
        im_transformed.save(name_transformed)
        

plt.imshow(full[4200], cmap='gray')
plt.imshow(full_transformed[6], cmap='gray')
