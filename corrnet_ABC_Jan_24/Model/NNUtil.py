

import os
import numpy as np
from scipy import sparse




def create_folder(folder):
	if not os.path.exists(folder):
		os.makedirs(folder)

def denseTheanoloader(file,  bit):
    mat = denseloader(file, bit)

    return mat
  
     


def sparseTheanoloader(file,  bit, row, col):
     mat = sparseloader(file, bit, row, col) 
     return mat

	


def denseloader(file, bit):
	# print "loading ...", file
	matrix = np.load(file + ".npy")
	matrix = np.array(matrix, dtype=bit)
	return matrix


def sparseloader(file, bit, row, col):
	print ("loading ...", file)
	x = np.load(file + "d.npy")
	y = np.load(file + "i.npy")
	z = np.load(file + "p.npy")
	matrix = sparse.csr_matrix((x, y, z), shape=(row, col), dtype=bit)
	matrix = matrix.todense()
	return matrix


