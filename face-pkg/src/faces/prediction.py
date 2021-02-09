#Libraries
import tensorflow as tf
import numpy as np
import os,glob,cv2
import sys,argparse

def person(image):
	image_size=200
	num_channels=3
	images = []

# The image is resized and preprocessed as done during training
	image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
	images.append(image)
	images = np.array(images, dtype=np.uint8)
	images = images.astype('float32')
	images = np.multiply(images, 1.0/255.0) 

# The input to the network is reshaped
	x_batch = images.reshape(1, image_size,image_size,num_channels)
# Restore the saved model 
	sess = tf.Session()
# Recreate the network graph
	saver = tf.train.import_meta_graph('/faces/face-rec-model.meta')
# Load the weights saved 
	saver.restore(sess, tf.train.latest_checkpoint('/faces/'))

# Accessing the default graph
	graph = tf.get_default_graph()

# y_pred (tensor) is the prediction of the network
	y_pred = graph.get_tensor_by_name("y_pred:0")
# Feeding the images to the input placeholders
	x = graph.get_tensor_by_name("x:0") 
	y = graph.get_tensor_by_name("y:0") 
	y_test_images = np.zeros((1, 15)) 

# Creating feed_dict that is required to calculate y_pred
	feed_dict_testing = {x: x_batch, y: y_test_images}
# Array with the possibilities of each class
	result=sess.run(y_pred, feed_dict=feed_dict_testing)

# Function to calculate the maximum of an array
	def max(array):

    		max = 0

    		for i in range(0,15):

        		if(array[0, i] > array[0, max]):

            			max = i

    		return max

# Function that returns the name associated to a integer
	def name(a):

		names = ["Name_1", "Name_2", "Name_3", "Name_4", "Name_5", "Name_6", 
        "Name_7", "Name_8", "Name_9", "Name_10", "Name_11", "Name_12", "Name_13", 
        "Name_14", "Nadie"]
	
		name = names[a]

		return name

	a = max(result)

	person = name(a)

	return person, result, a
