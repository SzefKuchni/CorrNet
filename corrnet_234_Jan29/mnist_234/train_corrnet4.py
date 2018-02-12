

import sys

sys.path.append("../Model/")
from corrnet_4 import trainCorrNet
from utils import get_setting_from_name_code

def start_train(src_folder,tgt_folder,size):
    src_folder = src_folder+"matpic1/"



    batch_size = 100
    training_epochs = 50 #int(sys.argv[3])
    l_rate = 0.01
    optimization = "rmsprop"
    tied = True
    
    n_hidden = 50 # 50/100
    lamda = 2
    hidden_activation = "sigmoid"
    output_activation = "sigmoid"
    loss_fn = "squarrederror"
    loss_type="1111"#str(sys.argv[4])# 2, 3
    
    #size = 252#14*14#252#392#252
    
    
    MODE_NUM, view1_name,view2_name = get_setting_from_name_code(tgt_folder)
    #print mode,MODE_NUM, view1_name, view2_name
    
    trainCorrNet(src_folder=src_folder, tgt_folder=tgt_folder, batch_size=batch_size,
                 training_epochs=training_epochs, l_rate=l_rate, optimization=optimization,
                 tied=tied, n_visible=size,
                 n_hidden=n_hidden, lamda=lamda, hidden_activation=hidden_activation,
                 output_activation=output_activation, loss_fn=loss_fn,loss_type=loss_type,
                 MODE_NUM=MODE_NUM,v1_name=view1_name,v2_name=view2_name)

if __name__=='__main__':
    src_folder = sys.argv[1]
    tgt_folder = sys.argv[2]
    size = 14*14#252#14*14#252
    start_train(src_folder,tgt_folder,size) #e.g. python train_corrnet4.py input/M_ABCD/ output/T_ABCD
