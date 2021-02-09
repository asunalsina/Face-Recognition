#!/usr/bin/env python

""" Node recognition.
This nodes implements the face recognition neural network.
"""

# Libraries
import prediction
import tensorflow as tf
import numpy as np
import os,glob,cv2
import rospy
from speech.srv import say
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge


class Recognition():
    def __init__(self):

	self.i = 0
        self._session = tf.Session()
        self._cv_bridge = CvBridge()
# Subscriber
        self._sub = rospy.Subscriber('face', Image, self.callback, queue_size=1)
# Publisher
        self._pub = rospy.Publisher('name', String, queue_size=1)

    def callback(self, ros_image):
# Loop to process an image when i%50==0. This avoid that all the images that node detection sends are processed. Node detection processes faster than node recognition.
	if (self.i)%50 == 0:
	  fresh = True
	  self.i += 1
	  cv_image = self._cv_bridge.imgmsg_to_cv2(ros_image, "bgr8")
	else:
	  fresh = False
	  self.i += 1

	if fresh == True:
	  
	  name, result, a = prediction.person(cv_image)
# String that is published
	  stt = name + ": " + str(result[0,a])
# Initialization of service say
          rospy.wait_for_service('speech/say_text')
          saytxt = rospy.ServiceProxy('speech/say_text', say)

          if name == "Nadie":
# Text that the robot says when no one is recognized
	   text = "Bienvenido. Espere mientras aviso a la anfitriona."
# Call of service say
	   saytxt(text)

	  else:
# Text that the robot says when someone is recognized
	    text = "Hola %s." % (name)
# Call of service say
	    saytxt(text)
 

	  cv2.imshow("face", cv_image)
	  cv2.waitKey(1000)
          self._pub.publish(stt)

    def main(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('recognition')
    tensor = Recognition()
tensor.main()
