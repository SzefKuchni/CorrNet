import matplotlib.pyplot as plt
model = CorrNet()
model.load(tgt_folder)


# weights from input to hidden layer
model.W_left.get_value()
model.W_left.get_value().shape
# (392, 50)
model.W_right.get_value()
model.W_right.get_value().shape
# (392, 50)


# weights from input to hidden layer
model.W_left_prime.eval()
model.W_left_prime.eval().shape
# (50, 392)
model.W_right_prime.eval()
model.W_right_prime.eval().shape
# (50, 392)

model.W_left.get_value()[0][0]

###########check for Theano
folder='C:\\Users\\T540pDLEWYNBQ\\Desktop\\CorrNet - Theano\\MNIST_DIR\\'
#input mnist image has shape fo 28 by 28 pixels which gives a vector of 784
mat1 = get_mat(folder+"train-view1.txt")
#this is half of the image so the vector is of lenght 392
mat2 = get_mat(folder+"train-view2.txt")


input=mat1[1]
plt.imshow(input.reshape(28,14), cmap='gray')

x_left=input
W_left=model.W_left.get_value()
b=model.b.get_value()
W_left_prime=model.W_left_prime.eval()
b_prime_left=model.b_prime_left.eval()

# reconstruction
y1_pre = T.dot(x_left, W_left) + b
y1 = activation(y1_pre, "sigmoid")

z1_left_pre = T.dot(y1, W_left_prime) + b_prime_left
z1_left = activation(z1_left_pre, "sigmoid")

result=z1_left.eval()
plt.imshow(result.reshape(28,14), cmap='gray')


################ check for Tensorflow
folder='C:\\Users\\T540pDLEWYNBQ\\Desktop\\CorrNet - Theano\\MNIST_DIR\\'
#input mnist image has shape fo 28 by 28 pixels which gives a vector of 784
mat1 = get_mat(folder+"train-view1.txt")
#this is half of the image so the vector is of lenght 392
mat2 = get_mat(folder+"train-view2.txt")


input=mat1[1]
plt.imshow(input.reshape(28,14), cmap='gray')

test_x_left=input
test_W_left=sess.run(W_left)
test_b=sess.run(b)
test_W_left_prime=sess.run(W_left_prime)
test_b_prime_left=sess.run(b_prime_left)

# reconstruction
y1_pre = tf.matmul(test_x_left.reshape(1,392), test_W_left) + test_b
y1 = tf.sigmoid(y1_pre)

z1_left_pre = tf.matmul(y1, test_W_left_prime) + test_b_prime_left
z1_left = tf.sigmoid(z1_left_pre)

result2=sess.run(z1_left)
plt.imshow(result2.reshape(28,14), cmap='gray')
