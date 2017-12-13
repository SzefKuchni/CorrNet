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
img=full[3]
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

bounding_square_size=max(width_max-width_min,height_max-height_min)+1
bounding_square_size

plt.imshow(img, cmap='gray')
img_cropped=img[height_min:height_min+bounding_square_size, width_min:width_min+bounding_square_size]
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

for i in range(0, 10000):
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

    bounding_square_size=max(width_max-width_min,height_max-height_min)+1
    
    img=scipy.misc.imresize(img,[28,28])
    img_cropped=img[height_min:height_min+bounding_square_size, width_min:width_min+bounding_square_size]
    img_cropped_resized=scipy.misc.imresize(img_cropped,[28,28])
    full_transformed[i]=img_cropped_resized
    #monitoring progress
    if i%100==0:
        print(i)
    #checking results
    if i<=10:
        im_original = Image.fromarray(img).convert("RGB")
        im_transformed = Image.fromarray(img_cropped_resized).convert("RGB")
        name_original="plots/image_"+str(i)+".png"
        name_transformed="plots/image_"+str(i)+"_transformed.png"
        im_original.save(name_original)
        im_transformed.save(name_transformed)
        

i=3000
plt.imshow(full[i], cmap='gray')
plt.imshow(full_transformed[i], cmap='gray')
