

import time
import pickle


from Initializer import tf_init_fan_sigmoid
from NNUtil import denseTheanoloader,sparseTheanoloader
from utils import get_view_name


import tensorflow as tf
import numpy as np
import sys
#sys.path.append("../input/")



class CorrNet(object):

    def init(self,  l_rate=0.01, optimization="sgd",
             tied=False, n_visible=None,n_hidden=None, lamda=5,
             hidden_activation="sigmoid", output_activation="sigmoid", loss_fn = "squarrederror",
             op_folder=None,MODE_NUM = 4):


        self.MODE_NUM=MODE_NUM
        self.l_rate = l_rate
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.lamda = lamda
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.loss_fn = loss_fn
        self.tied = tied
        self.op_folder = op_folder
        self.params={}
        self.input={}
######parameters and input
        for ii in range(self.MODE_NUM):
            view_name=str(ii+1)
            self.params["W1"+view_name] =tf_init_fan_sigmoid(n_visible,n_hidden,'W1'+view_name)
            if not tied:
                self.params['W2'+view_name] = tf_init_fan_sigmoid(n_hidden,n_visible,'W2'+view_name)
            else:
                self.params["W2"+view_name] = tf.transpose(self.params["W1"+view_name])
            self.params['b2'+view_name] = tf.Variable(tf.zeros([1, n_visible],tf.float32),name = 'b2'+view_name)
            self.input['x'+view_name]=tf.placeholder(tf.float32,[None,n_visible],name ='x'+view_name)

        self.params['b'] = tf.Variable(tf.zeros([1,n_hidden],tf.float32),name = 'b')

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
        L1=0
        for ii in range(self.MODE_NUM):
            view_name=str(ii+1)
            z1_left=self.one_layer(y1,self.params['W2'+view_name],self.params['b2'+view_name])
            L1= L1 + tf.reduce_sum(tf.pow(self.input['x'+view_name]-z1_left,2),1)
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
        L4 = tf.reduce_sum(corr) 
        return L4

    def train_common(self,mtype="1111"):
        
        res_loss=0
        corr_loss=0
        y={}
        for ii in range(self.MODE_NUM):
            view_name=str(ii+1)
            y[view_name]=self.one_layer(self.input['x'+view_name],self.params['W1'+view_name],self.params['b'])
            res_loss = res_loss + self.output_path(y[view_name])
            
        for ii in range(self.MODE_NUM):
            for j in range(ii+1,self.MODE_NUM):
                v1=str(ii+1)
                v2=str(j+1)
                corr_loss = corr_loss + self.corr_path(y[v1],y[v2])
                
#        y_all=0
#        for ii in range(self.MODE_NUM):
#            y_all= y_all + self.one_layer(self.input['x'+view_name],self.params['W1'+view_name],0)
#        y_all = y_all + self.params['b']
#        res_loss = res_loss + self.output_path(y_all)
#        y_ab_pre = tf.matmul(self.x_left,self.W_left)+tf.matmul(self.x_middle,self.W_middle)+self.b
#        y_ab = tf.nn.sigmoid(y_ab_pre)
#        L_ab=self.output_path(y_ab)
#        
#        y_ac_pre = tf.matmul(self.x_left,self.W_left)+tf.matmul(self.x_right,self.W_right)+self.b
#        y_ac = tf.nn.sigmoid(y_ac_pre)
#        L_ac=self.output_path(y_ac)
#        
#        y_bc_pre =tf.matmul(self.x_right,self.W_right)+tf.matmul(self.x_middle,self.W_middle)+self.b
#        y_bc = tf.nn.sigmoid(y_bc_pre)
#        L_bc=self.output_path(y_bc)

         
#        ly4_pre = tf.add(tf.matmul(self.x_left, self.W_left),self.b)
#        ly4 = tf.nn.sigmoid(ly4_pre)
#        lz4_right_pre = tf.add(tf.matmul(ly4,self.W_right_prime),self.b_prime_right)
#        lz4_right = tf.nn.sigmoid(lz4_right_pre)
#        ry4_pre = tf.add(tf.matmul(self.x_right, self.W_right),self.b)
#        ry4 = tf.nn.sigmoid(ry4_pre)
#        rz4_left_pre = tf.add(tf.matmul(ry4,self.W_left_prime),self.b_prime_left)
#        rz4_left = tf.nn.sigmoid(rz4_left_pre)
#        L5=tf.reduce_sum(tf.pow((lz4_right-self.x_right),2),1)+tf.reduce_sum(tf.pow((self.x_left-rz4_left),2),1)
##         
#        #L=L11+L22+L33-L4
#
#        if mtype=="1111":
#            print ("1111")
#            L = #L12+L13+L21+L23+L31+L32-L4+L11+L22+L33#+L_ab+L_bc+L_ac
#
#        elif mtype=="1":
#            print ("1")
#            L = L1 + L2 + L6 -L4
#
#        elif mtype=="3":
#            print ("3")
#            L = L1 + L2 + L6 +L3 - L4+L_ab+L_bc+L_ac

        L=-corr_loss*self.lamda+res_loss
        cost = tf.reduce_mean(L)
        optm = tf.train.RMSPropOptimizer(self.l_rate).minimize(cost)
        
        return (cost,optm)


#
#    def project_from_left(self,mat):
#        y_pre=tf.add(tf.matmul(mat,self.W_left),self.b)
#        y=tf.nn.sigmoid(y_pre)
#        return y
#
#    def project_from_right(self,mat):
#        y_pre=tf.add(tf.matmul(mat,self.W_right),self.b)
#        y=tf.nn.sigmoid(y_pre)
#        return y
#    def project_from_middle(self,mat):
#        y_pre=tf.add(tf.matmul(mat,self.W_middle),self.b)
#        y=tf.nn.sigmoid(y_pre)
#        return y
#    def project_r_from_middle(self,mat):
#        left,right,middle=self.reconstruct_from_middle(mat)
#        #y = self.project_from_left(left)
#        return left
#    def project_r_from_right(self,mat):
#        left,right,middle=self.reconstruct_from_right(mat)
#        #y = self.project_from_left(left)
#        return left
    def reconstruct(self,mat,view_num): # give one view to recover whole pic
        view_name=str(view_num)
        y= self.one_layer(mat,self.params["W1"+view_name],self.params['b'])
        output=[]
        for ii in range(self.MODE_NUM):
            tgt_view_name=str(ii+1)
            z = self.one_layer(y,self.params["W2"+tgt_view_name],self.params['b2'+tgt_view_name])
            output.append(z)
        return output
    

    def get_lr_rate(self):
        return self.optimizer.get_l_rate()

    def set_lr_rate(self,new_lr):
        self.optimizer.set_l_rate(new_lr)

    def save_matrices(self):
        for key,value in self.params.items():
            np.save(self.op_folder+key, value)

#        for p,nm in zip(self.params, self.param_names):
#            np.save(self.op_folder+nm, p)#p.get_value(borrow=True)
            

    def save_params(self):

        params = {}
        params["l_rate"] = self.l_rate
        params['n_visible']=self.n_visible
#        params["n_visible_left"] = self.n_visible_left
#        params["n_visible_right"] = self.n_visible_right
#        params["n_visible_middle"] = self.n_visible_middle
        params["n_hidden"] = self.n_hidden
        params["lamda"] = self.lamda
        params["hidden_activation"] = self.hidden_activation
        params["output_activation"] = self.output_activation
        params["tied"] = self.tied
        params["MODE_NUM"]=self.MODE_NUM


        pickle.dump(params,open(self.op_folder+"params.pck","wb"),-1)




def trainCorrNet(src_folder, tgt_folder, batch_size = 20, training_epochs=40,
                 l_rate=0.01, optimization="sgd", tied=False, lamda=5,n_visible=None,n_hidden=None,

                 hidden_activation="sigmoid",output_activation="sigmoid", loss_fn = "squarrederror",loss_type="1111",
                 MODE_NUM=4,v1_name="1",v2_name="2"):


    

    model = CorrNet()
    model.init( l_rate=l_rate, optimization=optimization, tied=tied,lamda=lamda,

    n_hidden=n_hidden, n_visible=n_visible, MODE_NUM = MODE_NUM,

    hidden_activation=hidden_activation, output_activation=output_activation,\
    loss_fn =loss_fn, op_folder=tgt_folder)
    

    start_time = time.clock()
     

    diff = 0
    flag = 1
    detfile = open(tgt_folder+"details.txt","w")
    detfile.close()
    #oldtc = float("inf")
    common_loss,common_opt=model.train_common(loss_type)#("1111")
    
    
    
    init_v = tf.global_variables_initializer()
    saver = tf.train.Saver()
    train_set={}
    batch={}
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
                        for ii in range(MODE_NUM):
                            view_name=get_view_name(ii,MODE_NUM,v1_name,v2_name)
                            if(next[1]=="dense"):
                                train_set[str(ii+1)]=denseTheanoloader(next[2]+"_"+view_name,"float32")
                            else:
                                train_set[str(ii+1)]=sparseTheanoloader(next[2]+"_"+view_name,"float32",1000,n_visible)
                  
                    
                    for index in range(0,int(float(next[3])/batch_size)):
                        for ii in range(MODE_NUM):
                            view_name=str(ii+1)
                            batch[view_name] =train_set[view_name][index * batch_size:(index + 1) * batch_size]
  
                        _,common_cost=sess.run([common_opt,common_loss],feed_dict={model.input["x"+str(rr+1)]: batch[str(rr+1)] for rr in range(MODE_NUM)})
                        c.append(common_cost)
                        
                        
#               

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

