import numpy as np
import tensorflow as tf

def get_mat(fname):
 
	file = open(fname,"r")
	mat = list()
	for line in file:
		line = line.strip().split()
		mat.append(line)
	mat = np.asarray(mat,dtype="float32")
	return mat


folder='C:\\Users\\T540pDLEWYNBQ\\Desktop\\CorrNet - Theano\\MNIST_DIR\\'
#input mnist image has shape fo 28 by 28 pixels which gives a vector of 784
img_left = get_mat(folder+"train-view1.txt")
img_left[0].shape
#this is half of the image so the vector is of lenght 392
img_right = get_mat(folder+"train-view2.txt")

L=50

X_left_ph = tf.placeholder(tf.float32, [None, 1, 392], name="left")
X_right_ph = tf.placeholder(tf.float32, [None, 1, 392], name="right")

XX_left_ph = tf.reshape(X_left_ph, [-1, 392])
XX_right_ph = tf.reshape(X_right_ph, [-1, 392])

W_left = tf.Variable(tf.truncated_normal([392, L], stddev=0.1))
W_right = tf.Variable(tf.truncated_normal([392, L], stddev=0.1))
b = tf.Variable(tf.zeros([L]))
b_prime_left = tf.Variable(tf.zeros([392]))
b_prime_right = tf.Variable(tf.zeros([392]))

W_left_prime = tf.transpose(W_left)
W_right_prime = tf.transpose(W_right)

y1 = tf.nn.sigmoid(tf.matmul(XX_left_ph, W_left) + b)
z1_left = tf.nn.sigmoid(tf.matmul(y1, W_left_prime) + b_prime_left)
z1_right = tf.nn.sigmoid(tf.matmul(y1, W_right_prime) + b_prime_right)

L1 = tf.squared_difference(z1_left, XX_left_ph) + tf.squared_difference(z1_right, XX_right_ph)
       
y2 = tf.nn.sigmoid(tf.matmul(XX_right_ph, W_right) + b)
z2_left = tf.nn.sigmoid(tf.matmul(y2, W_left_prime) + b_prime_left)
z2_right = tf.nn.sigmoid(tf.matmul(y2, W_right_prime) + b_prime_right)

L2 = tf.squared_difference(z2_left, XX_left_ph) + tf.squared_difference(z2_right, XX_right_ph)

y3 = tf.nn.sigmoid(tf.matmul(XX_right_ph, W_right) + tf.matmul(XX_left_ph, W_left) + b)
z3_left = tf.nn.sigmoid(tf.matmul(y3, W_left_prime) + b_prime_left)
z3_right = tf.nn.sigmoid(tf.matmul(y3, W_right_prime) + b_prime_right)

L3 = tf.squared_difference(z3_left, XX_left_ph) + tf.squared_difference(z3_right, XX_right_ph)

y1_mean = tf.reduce_mean(y1)
y1_centered = y1 - y1_mean        
y2_mean = tf.reduce_mean(y2)
y2_centered = y2 - y2_mean
corr_nr = tf.reduce_sum(y1_centered * y2_centered)
corr_dr1 = tf.sqrt(tf.reduce_sum(y1_centered * y1_centered)+1e-8)
corr_dr2 = tf.sqrt(tf.reduce_sum(y2_centered * y2_centered)+1e-8)
corr_dr = corr_dr1 * corr_dr2
corr = corr_nr/corr_dr

L4 = tf.reduce_sum(corr) * 2

L=L1 + L2 + L3 - L4 

cost = tf.reduce_mean(L)

#####
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
print(sess.run(cost, feed_dict={X_left_ph:img_left[1:3].reshape(-1,1,392), X_right_ph:img_right[1:3].reshape(-1,1,392)}))

#####
train_step = tf.train.AdamOptimizer(0.01).minimize(cost)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(500):
    sess.run(train_step, feed_dict={X_left_ph:img_left[i*100:(i+1)*100].reshape(-1,1,392), X_right_ph:img_right[i*100:(i+1)*100].reshape(-1,1,392)})
    print(i)
    print(sess.run(W_left[0][0]))
    print(sess.run(cost, feed_dict={X_left_ph:img_left[i*100:(i+1)*100].reshape(-1,1,392), X_right_ph:img_right[i*100:(i+1)*100].reshape(-1,1,392)}))


######
epochs=50
batch_size=100
train_step = tf.train.AdamOptimizer(0.01).minimize(cost)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
    
    
for epoch in range(epochs):
    avg_cost = 0
    total_batch = int(img_left.shape[0]/batch_size)
    for i in range(total_batch-1):
        _, c = sess.run([train_step, cost], feed_dict = {X_left_ph: img_left[i*100:(i+1)*100].reshape(-1,1,392), X_right_ph: img_right[i*100:(i+1)*100].reshape(-1,1,392)})
        avg_cost += c / total_batch
         
    print("Epoch:", (epoch+1), "cost =", "{:.5f}".format(avg_cost))
   
print("\nTraining complete!")