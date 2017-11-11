

import numpy as np
import tensorflow as tf


def tf_init_fan_sigmoid(d1,d2,nameStr):
    
    low = -1* np.sqrt(6./(d1+d2))
    high = 1* np.sqrt(6./(d1+d2))
    return tf.Variable(tf.random_uniform([d1,d2],low,high),name=nameStr)

	