__author__ = 'Sarath'

import numpy
import os
import sys
import tensorflow as tf
import pickle

sys.path.append("../Model/")
from corrnet_yaqian import CorrNet

def create_folder(folder):

	if not os.path.exists(folder):
		os.makedirs(folder)



src_folder = sys.argv[1]+"matpic/"
tgt_folder = sys.argv[2]

model = CorrNet()

folder=tgt_folder
plist = pickle.load(open(folder+"params.pck","rb"))


model.init(l_rate=plist["l_rate"],
          tied=plist["tied"],
          n_visible_left=plist["n_visible_left"], n_visible_right=plist["n_visible_right"],
          n_hidden=plist["n_hidden"], lamda=plist["lamda"], W_left=folder+"W_left",
          W_right=folder+"W_right", b=folder+"b", W_left_prime=folder+"W_left_prime",
          W_right_prime=folder+"W_right_prime", b_prime_left=folder+"b_prime_left",
          b_prime_right=folder+"b_prime_right", 
          hidden_activation=plist["hidden_activation"], output_activation=plist["output_activation"],
           op_folder=folder)

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess,tgt_folder)

    create_folder(tgt_folder+"project/")
    
    mat = numpy.load(src_folder+"train/view1.npy")
    new_mat = model.project_from_left(mat)
    numpy.save(tgt_folder+"project/train-view1",new_mat.eval())

    mat = numpy.load(src_folder+"train/view2.npy")
    new_mat = model.project_from_right(mat)
    numpy.save(tgt_folder+"project/train-view2",new_mat.eval())
    
    mat = numpy.load(src_folder+"train/labels.npy")
    numpy.save(tgt_folder+"project/train-labels",mat)
    
    
    mat = numpy.load(src_folder+"valid/view1.npy")
    new_mat = model.project_from_left(mat)
    numpy.save(tgt_folder+"project/valid-view1",new_mat.eval())
    
    mat = numpy.load(src_folder+"valid/view2.npy")
    new_mat = model.project_from_right(mat)
    numpy.save(tgt_folder+"project/valid-view2",new_mat.eval())
    
    mat = numpy.load(src_folder+"valid/labels.npy")
    numpy.save(tgt_folder+"project/valid-labels",mat)
    
    
    mat = numpy.load(src_folder+"test/view1.npy")
    new_mat = model.project_from_left(mat)
    numpy.save(tgt_folder+"project/test-view1",new_mat.eval())
    
    mat = numpy.load(src_folder+"test/view2.npy")
    new_mat = model.project_from_right(mat)
    numpy.save(tgt_folder+"project/test-view2",new_mat.eval())
    
    mat = numpy.load(src_folder+"test/labels.npy")
    numpy.save(tgt_folder+"project/test-labels",mat)

