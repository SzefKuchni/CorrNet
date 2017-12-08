import os

input_dir_LR='CorrNet-py3.5\\mnistExample\\'

print('LEFT RIGHT experiment')
os.chdir(input_dir_LR)
print('Current directory: ',os.getcwd())

for x in range(0, 20):
	print('ITERATION: ', x)
	print('TRAINING')
	os.system('python train_corrnet.py MNIST_DIR/ TGT_DIR/')
	print('PROJECTING')
	os.system('python project_corrnet.py MNIST_DIR/ TGT_DIR/')
	print('TRANSFER LERARNING')
	os.system('python evaluate.py tl TGT_DIR/')
	print('CORRELATION')
	os.system('python evaluate.py corr TGT_DIR/')