import cv2
import os
import glob
from sklearn.utils import shuffle
import numpy as np


def load_train(t_path, image_size, people):
    images = []
    labels = []
    img_names = []
    cls = []
# Load and preprocess of the images
    print('Reading training images.')
    for fields in people:   
        index = people.index(fields)
        print('Reading {} files (Index: {})'.format(fields, index))
        path = os.path.join(t_path, fields, '*g')
        files = glob.glob(path)
        for fl in files:
            image = cv2.imread(fl)
            image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
            image = image.astype(np.float32)
            image = np.multiply(image, 1.0 / 255.0)
            images.append(image)
# To improve the dataset the images can be flipped horizontally and vertically
	    #image = cv2.flip(image, 0)
	    #image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
            #image = image.astype(np.float32)
            #image = np.multiply(image, 1.0 / 255.0)
            #images.append(image)
       	    label = np.zeros(len(people))
            label[index] = 1.0
            labels.append(label)
            flbase = os.path.basename(fl)
            img_names.append(flbase)
            cls.append(fields)
    images = np.array(images)
    labels = np.array(labels)
    img_names = np.array(img_names)
    cls = np.array(cls)

    return images, labels, img_names, cls


class DataSet(object):

  def __init__(self, images, labels, img_names, cls):
    self._num_examples = images.shape[0]

    self._images = images
    self._labels = labels
    self._img_names = img_names
    self._cls = cls
    self._epochs_done = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def img_names(self):
    return self._img_names

  @property
  def cls(self):
    return self._cls

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_done(self):
    return self._epochs_done

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size

    if self._index_in_epoch > self._num_examples:
      # After each epoch we update this
      self._epochs_done += 1
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch

    return self._images[start:end], self._labels[start:end], self._img_names[start:end], self._cls[start:end]


def read_train_sets(t_path, image_size, people, eval_size):
  class DataSets(object):
    pass
  data_sets = DataSets()

  images, labels, img_names, cls = load_train(t_path, image_size, people)
  images, labels, img_names, cls = shuffle(images, labels, img_names, cls)  

  if isinstance(eval_size, float):
    eval_size = int(eval_size * images.shape[0])

  eval_images = images[:eval_size]
  eval_labels = labels[:eval_size]
  eval_img_names = img_names[:eval_size]
  eval_cls = cls[:eval_size]

  train_images = images[eval_size:]
  train_labels = labels[eval_size:]
  train_img_names = img_names[eval_size:]
  train_cls = cls[eval_size:]

  data_sets.train = DataSet(train_images, train_labels, train_img_names, train_cls)
  data_sets.eval = DataSet(eval_images, eval_labels, eval_img_names, eval_cls)

  return data_sets


