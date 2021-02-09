# Libraries
import dataset
import tensorflow as tf
import math
import random
import numpy as np
from numpy.random import seed
from tensorflow import set_random_seed

seed(1)
set_random_seed(2)

batch_size = 128

# People that the network is going to recognise
people = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11', 'p12', 'p13', 'p14', 'p15']
num_people = len(people)

# Evaluation images will be the 25% of the data
eval_size = 0.25
image_size = 200
channels = 3
path = 'dataset'

# Load of the training and evaluation dataset.
data = dataset.read_train_sets(path, image_size, people, eval_size=eval_size)

print("Reading input data: completed.")
print("Files in training set:\t\t{}".format(len(data.train.labels)))
print("Files in evaluation set:\t{}".format(len(data.eval.labels)))

session = tf.Session()
x = tf.placeholder(tf.float32, shape=[None, image_size, image_size, channels], name='x')

# Labels
y = tf.placeholder(tf.float32, shape=[None, num_people], name='y')
y_cls = tf.argmax(y, axis=1)

# Layers of the neural network
# Convolutional layer
conv1 = tf.layers.conv2d(
        inputs=x,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
# Max pooling layer (200/2=100)
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
# Convolutional layer
conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
# Max pooling layer (100/2=50)
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
# Convolutional layer (64 filter)
conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
# Convolutional layer (50/2=25)
pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
# 25*25*64
pool3_flat = tf.reshape(pool3, [-1, 25*25*64])
# First dense layer with 1024 neurons
dense = tf.layers.dense(inputs=pool3_flat, units=1024, activation=tf.nn.relu)
# Dropout layer with a rate of 0.5. This means that if training is set to True this layer will drop half of the input
dropout = tf.layers.dropout(inputs=dense, rate=0.5, training=False)
# Second dense layer with the same number of neurons as classes on the dataset
logits = tf.layers.dense(inputs=dropout, units=num_people)

y_pred = tf.nn.softmax(logits, name='y_pred')

y_pred_cls = tf.argmax(y_pred, axis=1)
session.run(tf.global_variables_initializer())
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y)
cost = tf.reduce_mean(cross_entropy)
# AdamOptimizer with a learning rate of 0.001
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_cls)
precision = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session.run(tf.global_variables_initializer())

# Function that shows how the training and evaluation evolve
def show_progress(epoch, feed_dict_train, feed_dict_eval, loss):
    pre = session.run(precision, feed_dict=feed_dict_train)
    eval_pre = session.run(precision, feed_dict=feed_dict_eval)
    msg = "Epoch {0} --- Training precision: {1:>6.1%}, Evaluation precision: {2:>6.1%}, Loss: {3:.3f}"
    print(msg.format(epoch + 1, pre, eval_pre, loss))


total_iterations = 0

saver = tf.train.Saver()


def train(num_iteration):
    global total_iterations

    for i in range(total_iterations,
                   total_iterations + num_iteration):

        x_batch, y_batch, _, cls_batch = data.train.next_batch(batch_size)
        x_eval_batch, y_eval_batch, _, eval_cls_batch = data.eval.next_batch(batch_size)

        feed_dict_tr = {x: x_batch,
                        y: y_batch}
        feed_dict_val = {x: x_eval_batch,
                         y: y_eval_batch}

        session.run(optimizer, feed_dict=feed_dict_tr)

        if i % int(data.train.num_examples / batch_size) == 0:
            loss = session.run(cost, feed_dict=feed_dict_val)
            epoch = int(i / int(data.train.num_examples / batch_size))

            show_progress(epoch, feed_dict_tr, feed_dict_val, loss)
# Save the session
            saver.save(session, 'face-rec-model')



train(num_iteration=500)
