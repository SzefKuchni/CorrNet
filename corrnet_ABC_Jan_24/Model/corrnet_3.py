

import time
import pickle


from Initializer import tf_init_fan_sigmoid
from NNUtil import denseTheanoloader,sparseTheanoloader


import tensorflow as tf
import numpy as np
import sys
sys.path.append("../input/")


class CorrNet(object):

    def init(self,  l_rate=0.01, optimization="sgd",
             tied=False, n_visible_left=None, n_visible_right=None, n_visible_middle=None,n_hidden=None, lamda=5,
             W_left=None, W_right=None,W_middle=None, b=None, W_left_prime=None, W_right_prime=None, W_middle_prime=None,
             b_prime_left=None, b_prime_right=None,b_prime_middle=None, input_left=None, input_right=None,input_middle=None,
             hidden_activation="sigmoid", output_activation="sigmoid", loss_fn = "squarrederror",
             op_folder=None):


        self.l_rate = l_rate
        self.n_visible_left = n_visible_left
        self.n_visible_right = n_visible_right
        self.n_visible_middle = n_visible_middle
        
        self.n_hidden = n_hidden
        self.lamda = lamda
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.loss_fn = loss_fn
        self.tied = tied
        self.op_folder = op_folder
        
######parameters wl wr wl' wr' b b'
        self.W_left = tf_init_fan_sigmoid(n_visible_left,n_hidden,'W_left')
        self.W_middle = tf_init_fan_sigmoid(n_visible_right,n_hidden,'W_middle')      
        self.W_right = tf_init_fan_sigmoid(n_visible_right,n_hidden,'W_right')
        
        if not tied:
            self.W_left_prime = tf_init_fan_sigmoid(n_hidden,n_visible_left,'W_left_prime')
            self.W_middle_prime = tf_init_fan_sigmoid(n_hidden,n_visible_left,'W_middle_prime')           
            self.W_right_prime = tf_init_fan_sigmoid(n_hidden,n_visible_right,'W_right_prime')

        else:
            self.W_left_prime = tf.transpose(self.W_left)
            self.W_right_prime = tf.transpose(self.W_right)
            self.W_middle_prime = tf.transpose(self.W_middle)

        self.b = tf.Variable(tf.zeros([1,n_hidden],tf.float32),name = 'b')

        self.b_prime_left = tf.Variable(tf.zeros([1, n_visible_left],tf.float32),name = 'b_prime_left')
        self.b_prime_middle = tf.Variable(tf.zeros([1, n_visible_left],tf.float32),name = 'b_prime_middle')
        self.b_prime_right = tf.Variable(tf.zeros([1, n_visible_right],tf.float32),name = 'b_prime_right')

#        
############ input

        if input_left is None:
            #self.x_left = T.matrix(name='x_left')
            self.x_left = tf.placeholder(tf.float32,[None,n_visible_left],name ='x_left')
        else:
            self.x_left = input_left

        if input_right is None:
            #self.x_right = T.matrix(name='x_right')
            self.x_right = tf.placeholder(tf.float32,[None,n_visible_right],name='x_right')
        else:
            self.x_right = input_right
            
        if input_middle is None:
        #self.x_right = T.matrix(name='x_right')
            self.x_middle = tf.placeholder(tf.float32,[None,n_visible_middle],name='x_middle')
        else:
            self.x_middle = input_middle


        if tied:
            self.params = [self.W_left, self.W_right, self.W_middle, self.b, self.b_prime_left, self.b_prime_right,self.b_prime_middle]
            self.param_names = ["W_left", "W_right", "W_middle","b", "b_prime_left", "b_prime_right","b_prime_middle"]
        else:
            self.params = [self.W_left, self.W_right, self.W_middle, self.b, self.b_prime_left, self.b_prime_right,self.b_prime_middle, self.W_left_prime, self.W_right_prime,self.W_middle_prime]
            self.param_names = ["W_left", "W_right", "W_middle","b", "b_prime_left", "b_prime_right", "b_prime_middle","W_left_prime", "W_right_prime","W_middle_prime"]


        self.save_params()
    def one_layer(self,x,W, b):
        y_pre= tf.add(tf.matmul(x,W),b)
        y= tf.nn.sigmoid(y_pre)
        return y
    def two_layer(self,x, W1, b1, W2,b2):
        y1 = self.one_layer(x,W1,b1)
        y2 = self.one_layer(y1,W2,b2)

        return y2
        
    def output_path(self,input):
        y1=input
        z1_left=self.one_layer(y1,self.W_left_prime,self.b_prime_left)
        z1_middle=self.one_layer(y1,self.W_middle_prime,self.b_prime_middle)
        z1_right=self.one_layer(y1,self.W_right_prime,self.b_prime_right)
        
#        z1_left_pre = tf.add(tf.matmul(y1,self.W_left_prime),self.b_prime_left)
#        z1_right_pre = tf.add(tf.matmul(y1,self.W_right_prime),self.b_prime_right)
#        z1_middle_pre = tf.add(tf.matmul(y1,self.W_middle_prime),self.b_prime_middle)
#        z1_left = tf.nn.sigmoid(z1_left_pre)
#        z1_right= tf.nn.sigmoid(z1_right_pre)
#        z1_middle= tf.nn.sigmoid(z1_middle_pre)
        L1 = tf.reduce_sum(tf.pow(self.x_left-z1_left,2),1)+tf.reduce_sum(tf.pow(self.x_right-z1_right,2),1)+tf.reduce_sum(tf.pow(self.x_middle-z1_middle,2),1)
        return L1
        
    def corr_path(self,y1,y2):
        y1_mean = tf.reduce_mean(y1, axis=0)
        y1_centered = y1 - y1_mean
        y2_mean = tf.reduce_mean(y2, axis=0)
        y2_centered = y2 - y2_mean
        corr_nr = tf.reduce_sum(tf.multiply(y1_centered,y2_centered),axis=0)#axis =0
        corr_dr1 = tf.sqrt(tf.reduce_sum(tf.multiply(y1_centered,y1_centered),axis=0)+1e-8)
        corr_dr2 = tf.sqrt(tf.reduce_sum(tf.multiply(y2_centered,y2_centered),axis=0)+1e-8)
        corr_dr = tf.multiply(corr_dr1, corr_dr2)
        corr = tf.div(corr_nr,corr_dr)
        L4 = tf.reduce_sum(corr) * self.lamda
        return L4

    def train_common(self,mtype="1111"):
        y1=self.one_layer(self.x_left,self.W_left,self.b)
        z1_left=self.one_layer(y1,self.W_left_prime,self.b_prime_left)
        z1_middle=self.one_layer(y1,self.W_middle_prime,self.b_prime_middle)
        z1_right=self.one_layer(y1,self.W_right_prime,self.b_prime_right)
        L11 = tf.reduce_sum(tf.pow(self.x_left-z1_left,2),1)
        L12 =tf.reduce_sum(tf.pow(self.x_middle-z1_middle,2),1)
        L13 =tf.reduce_sum(tf.pow(self.x_right-z1_right,2),1)
       
          
        y2 = self.one_layer(self.x_middle,self.W_middle,self.b)
        z2_left=self.one_layer(y2,self.W_left_prime,self.b_prime_left)
        z2_middle=self.one_layer(y2,self.W_middle_prime,self.b_prime_middle)
        z2_right=self.one_layer(y2,self.W_right_prime,self.b_prime_right)
        L21 = tf.reduce_sum(tf.pow(self.x_left-z2_left,2),1)
        L22 = tf.reduce_sum(tf.pow(self.x_middle-z2_middle,2),1)
        L23 = tf.reduce_sum(tf.pow(self.x_right-z2_right,2),1)
       
 
        y3 = self.one_layer(self.x_right,self.W_right,self.b)
        z3_left=self.one_layer(y3,self.W_left_prime,self.b_prime_left)
        z3_middle=self.one_layer(y3,self.W_middle_prime,self.b_prime_middle)
        z3_right=self.one_layer(y3,self.W_right_prime,self.b_prime_right)
        L31 = tf.reduce_sum(tf.pow(self.x_left-z3_left,2),1)
        L32 = tf.reduce_sum(tf.pow(self.x_middle-z3_middle,2),1)
        L33 = tf.reduce_sum(tf.pow(self.x_right-z3_right,2),1)
        
       
        
        y_ab_pre = tf.matmul(self.x_left,self.W_left)+tf.matmul(self.x_middle,self.W_middle)+self.b
        y_ab = tf.nn.sigmoid(y_ab_pre)
        L_ab=self.output_path(y_ab)
        
        y_ac_pre = tf.matmul(self.x_left,self.W_left)+tf.matmul(self.x_right,self.W_right)+self.b
        y_ac = tf.nn.sigmoid(y_ac_pre)
        L_ac=self.output_path(y_ac)
        
        y_bc_pre =tf.matmul(self.x_right,self.W_right)+tf.matmul(self.x_middle,self.W_middle)+self.b
        y_bc = tf.nn.sigmoid(y_bc_pre)
        L_bc=self.output_path(y_bc)
        
        L4=self.corr_path(y1,y2)+self.corr_path(y1,y3)+self.corr_path(y2,y3)
#        y1_mean = tf.reduce_mean(y1, axis=0)
#        y1_centered = y1 - y1_mean
#        y2_mean = tf.reduce_mean(y2, axis=0)
#        y2_centered = y2 - y2_mean
#        corr_nr = tf.reduce_sum(tf.multiply(y1_centered,y2_centered),axis=0)#axis =0
#        corr_dr1 = tf.sqrt(tf.reduce_sum(tf.multiply(y1_centered,y1_centered),axis=0)+1e-8)
#        corr_dr2 = tf.sqrt(tf.reduce_sum(tf.multiply(y2_centered,y2_centered),axis=0)+1e-8)
#        corr_dr = tf.multiply(corr_dr1, corr_dr2)
#        corr = tf.div(corr_nr,corr_dr)
#        L4 = tf.reduce_sum(corr) * self.lamda
         
        ly4_pre = tf.add(tf.matmul(self.x_left, self.W_left),self.b)
        ly4 = tf.nn.sigmoid(ly4_pre)
        lz4_right_pre = tf.add(tf.matmul(ly4,self.W_right_prime),self.b_prime_right)
        lz4_right = tf.nn.sigmoid(lz4_right_pre)
        ry4_pre = tf.add(tf.matmul(self.x_right, self.W_right),self.b)
        ry4 = tf.nn.sigmoid(ry4_pre)
        rz4_left_pre = tf.add(tf.matmul(ry4,self.W_left_prime),self.b_prime_left)
        rz4_left = tf.nn.sigmoid(rz4_left_pre)
        L5=tf.reduce_sum(tf.pow((lz4_right-self.x_right),2),1)+tf.reduce_sum(tf.pow((self.x_left-rz4_left),2),1)
         
        L=L11+L22+L33-L4

        if mtype=="1111":
            print ("1111")
            L = L+L12+L13+L21+L23+L31+L23

        elif mtype=="1":
            print ("1")
            L = L1 + L2 + L6 -L4
        elif mtype=="2":
            print ("2")
            L = L1 + L2 + L6 +L3 - L4
        elif mtype=="3":
            print ("3")
            L = L1 + L2 + L6 +L3 - L4+L_ab+L_bc+L_ac

        elif mtype=="1101":
            print ("1101")
            L = L1 + L2 - L4

        elif mtype == "0011":
            #print "0011"
            print ("0011")
            L = L3 - L4
        elif mtype == "1100":
            #print "1100"
            print ("1100")
            L = L1 + L2
        elif mtype == "1100":
            #print "0010"
            print ("0010")
            L = L3
        elif mtype == "euc":
            #print "euc"
            print ("euc")
            L = L5
        elif mtype == "euc-cor":
            print ("euc-cor")
            L = L5 - L4

        cost = tf.reduce_mean(L)
        optm = tf.train.RMSPropOptimizer(self.l_rate).minimize(cost)
        
        return (cost,optm)

    def train_left(self):
        y_pre = tf.add(tf.matmul(self.x_left, self.W_left) , self.b)
        y = tf.nn.sigmoid(y_pre)
        z_left_pre = tf.add(tf.matmul(y, self.W_left_prime) , self.b_prime_left)
        z_left = tf.nn.sigmoid(z_left_pre)
        L =tf.reduce_sum(tf.pow(z_left-self.x_left,2),1)
        cost = tf.reduce_mean(L)
        optm = tf.train.RMSPropOptimizer(self.l_rate).minimize(cost)

        return (cost, optm)


    def train_right(self):

        y_pre = tf.add(tf.matmul(self.x_right, self.W_right) , self.b)
        y = tf.nn.sigmoid(y_pre)
        z_right_pre =tf.add(tf.matmul(y, self.W_right_prime) ,self.b_prime_right)
        z_right = tf.nn.sigmoid(z_right_pre)
        L = tf.reduce_sum(tf.pow(z_right-self.x_right, 2),1)
        cost = tf.reduce_mean(L)
        optm = tf.train.RMSPropOptimizer(self.l_rate).minimize(cost)
        return (cost, optm)

    def project_from_left(self,mat):
        y_pre=tf.add(tf.matmul(mat,self.W_left),self.b)
        y=tf.nn.sigmoid(y_pre)
        return y

    def project_from_right(self,mat):
        y_pre=tf.add(tf.matmul(mat,self.W_right),self.b)
        y=tf.nn.sigmoid(y_pre)
        return y
    def project_from_middle(self,mat):
        y_pre=tf.add(tf.matmul(mat,self.W_middle),self.b)
        y=tf.nn.sigmoid(y_pre)
        return y
    def project_r_from_middle(self,mat):
        left,right,middle=self.reconstruct_from_middle(mat)
        #y = self.project_from_left(left)
        return left
    def project_r_from_right(self,mat):
        left,right,middle=self.reconstruct_from_right(mat)
        #y = self.project_from_left(left)
        return left
    def reconstruct(self,y):
        z_left_pre = tf.add(tf.matmul(y,self.W_left_prime),self.b_prime_left)
        z_right_pre = tf.add(tf.matmul(y,self.W_right_prime),self.b_prime_right)
        z_middle_pre = tf.add(tf.matmul(y,self.W_middle_prime),self.b_prime_middle)
        z_left = tf.nn.sigmoid(z_left_pre)
        z_right = tf.nn.sigmoid(z_right_pre)
        z_middle = tf.nn.sigmoid(z_middle_pre)
        return z_left, z_right,z_middle
        
    def reconstruct_from_left(self,mat):
        y_pre=tf.add(tf.matmul(mat,self.W_left),self.b)
        y= tf.nn.sigmoid(y_pre)
        z_left,z_right,z_middle=self.reconstruct(y)
        return z_left,z_right,z_middle
#        z_left_pre = tf.add(tf.matmul(y,self.W_left_prime),self.b_prime_left)
#        z_right_pre = tf.add(tf.matmul(y,self.W_right_prime),self.b_prime_right)
#        z_middle_pre = tf.add(tf.matmul(y,self.W_middle_prime),self.b_prime_middle)
#        z_left = tf.nn.sigmoid(z_left_pre)
#        z_right = tf.nn.sigmoid(z_right_pre)
#        z_middle = tf.nn.sigmoid(z_middle_pre)
#        return z_left, z_right

    def reconstruct_from_right(self,mat):
        y_pre=tf.add(tf.matmul(mat,self.W_right),self.b)
        y= tf.nn.sigmoid(y_pre)
        z_left,z_right,z_middle=self.reconstruct(y)
        return z_left,z_right,z_middle
#        z_left_pre = tf.add(tf.matmul(y,self.W_left_prime),self.b_prime_left)
#        z_right_pre = tf.add(tf.matmul(y,self.W_right_prime),self.b_prime_right)
#        z_left = tf.nn.sigmoid(z_left_pre)
#        z_right = tf.nn.sigmoid(z_right_pre)
#
#        return z_left, z_right
    def reconstruct_from_middle(self,mat):
        y_pre=tf.add(tf.matmul(mat,self.W_middle),self.b)
        y= tf.nn.sigmoid(y_pre)
        z_left,z_right,z_middle=self.reconstruct(y)
        return z_left,z_right,z_middle

    def get_lr_rate(self):
        return self.optimizer.get_l_rate()

    def set_lr_rate(self,new_lr):
        self.optimizer.set_l_rate(new_lr)

    def save_matrices(self):

        for p,nm in zip(self.params, self.param_names):
            np.save(self.op_folder+nm, p)#p.get_value(borrow=True)
            

    def save_params(self):

        params = {}
        params["l_rate"] = self.l_rate
        params["n_visible_left"] = self.n_visible_left
        params["n_visible_right"] = self.n_visible_right
        params["n_visible_middle"] = self.n_visible_middle
        params["n_hidden"] = self.n_hidden
        params["lamda"] = self.lamda
        params["hidden_activation"] = self.hidden_activation
        params["output_activation"] = self.output_activation
        params["tied"] = self.tied


        pickle.dump(params,open(self.op_folder+"params.pck","wb"),-1)




def trainCorrNet(src_folder, tgt_folder, batch_size = 20, training_epochs=40,
                 l_rate=0.01, optimization="sgd", tied=False, n_visible_left=None,
                 n_visible_right=None,n_visible_middle=None,  n_hidden=None, lamda=5,
                 W_left=None, W_right=None,W_middle=None, b=None, W_left_prime=None, W_right_prime=None,W_middle_prime=None,
                 b_prime_left=None, b_prime_right=None,b_prime_middle=None, hidden_activation="sigmoid",
                 output_activation="sigmoid", loss_fn = "squarrederror",loss_type="1111"):


    x_left = None
    x_right = None
    x_middle = None

    

    model = CorrNet()
    model.init( l_rate=l_rate, optimization=optimization, tied=tied,\
    n_visible_left=n_visible_left, n_visible_right=n_visible_right,n_visible_middle=n_visible_middle, \
    n_hidden=n_hidden, lamda=lamda, W_left=W_left, W_right=W_right,W_middle=W_middle, \
    b=b, W_left_prime=W_left_prime, W_right_prime=W_right_prime,W_middle_prime=W_middle_prime, \
    b_prime_left=b_prime_left, b_prime_right=b_prime_right,b_prime_middle=b_prime_middle, \
    input_left=x_left, input_right=x_right,input_middle=x_middle, hidden_activation=hidden_activation, output_activation=output_activation,\
    loss_fn =loss_fn, op_folder=tgt_folder)
    

    start_time = time.clock()
    
    train_set_x_left = np.asarray(np.zeros((1000,n_visible_left)), dtype=np.float32)
    train_set_x_right = np.asarray(np.zeros((1000,n_visible_right)), dtype=np.float32)
    train_set_x_middle = np.asarray(np.zeros((1000,n_visible_middle)), dtype=np.float32)


#   

    diff = 0
    flag = 1
    detfile = open(tgt_folder+"details.txt","w")
    detfile.close()
    #oldtc = float("inf")
    common_loss,common_opt=model.train_common(loss_type)#("1111")
    left_loss,left_opt=model.train_left()
    right_loss,right_opt=model.train_right()
    middle_loss,middle_opt=model.train_right()
    
    
    
    init_v = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init_v)
        #training cycle
        for epoch in range(training_epochs):
    
            print ("in epoch ", epoch)
            c = []
            ipfile = open(src_folder+"train/ip.txt","r")
            for line in ipfile:
                next = line.strip().split(",")
                if(next[0]=="xy"):
                    if(next[1]=="dense"):
                        train_set_x_left=denseTheanoloader(next[2]+"_left","float32")
                        train_set_x_middle=denseTheanoloader(next[2]+"_middle","float32")
                        train_set_x_right = denseTheanoloader(next[2] + "_right", "float32")
                        
                    else:
                        train_set_x_left=sparseTheanoloader(next[2]+"_left","float32",1000,n_visible_left)
                        train_set_x_middle = sparseTheanoloader(next[2] + "_middle", "float32", 1000, n_visible_right)
                        train_set_x_right=sparseTheanoloader(next[2]+"_right","float32", 1000, n_visible_right)
                    
                    for index in range(0,int(float(next[3])/batch_size)):
                        batch_a =train_set_x_left[index * batch_size:(index + 1) * batch_size]
                        batch_b =train_set_x_middle[index * batch_size:(index + 1) * batch_size]
                        batch_c = train_set_x_right[index * batch_size:(index + 1) * batch_size]
                        
                        _,common_cost=sess.run([common_opt,common_loss],feed_dict={model.x_left:batch_a,model.x_middle:batch_b,model.x_right:batch_c})
                        c.append(common_cost)
                        
                        
#                elif(next[0]=="x"):
#                    if(next[1]=="dense"):
#                        train_set_x_left=denseTheanoloader(next[2]+"_left","float32")
#                    else:
#                        train_set_x_left=sparseTheanoloader(next[2]+"_left","float32",1000,n_visible_left)
#                    for index in range(0,int(next[3])/batch_size):
#                        batch_x =train_set_x_left[index * batch_size:(index + 1) * batch_size]
#                        batch_y =np.zeros(np.shape(batch_x))
#                        _,left_cost=sess.run([left_opt,left_loss],feed_dict={model.x_left:batch_x,model.x_right:batch_y})
#                        c.append(left_cost)
#                        
#                elif(next[0]=="y"):
#                    if(next[1]=="dense"):
#                        train_set_x_right=denseTheanoloader(next[2]+"_right","float32")
#                    else:
#                        train_set_x_right=sparseTheanoloader(next[2]+"_right","float32",1000,n_visible_right)
#                    for index in range(0,int(next[3])/batch_size):
#                        
#                        batch_y =train_set_x_right[index * batch_size:(index + 1) * batch_size]
#                        batch_x =np.zeros(np.shape(batch_y))
#                        _,right_cost=sess.run([right_opt,right_loss],feed_dict={model.x_left:batch_x,model.x_right:batch_y})
#                        c.append(right_cost)


            if(flag==1):
                flag = 0
                diff = np.mean(c)
                di = diff
            else:
                di = np.mean(c) - diff
                diff = np.mean(c)
    
            print ('Difference between 2 epochs is ', di)
            print ('Training epoch %d, cost ' % epoch, diff)
    
            ipfile.close()
    
            detfile = open(tgt_folder+"details.txt","a")
            detfile.write("train\t"+str(diff)+"\n")
            detfile.close()

    

        end_time = time.clock()
        training_time = (end_time - start_time)
        print (' code ran for %.2fm' % (training_time / 60.))
        save_path = saver.save(sess, model.op_folder)
        print("Model saved in file: %s" % save_path)

