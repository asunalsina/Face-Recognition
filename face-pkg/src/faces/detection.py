#!/usr/bin/env python

""" Node detection.
This node implements the algorithm for face detection.
"""

# Libraries
import detector
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

class Detection():

    def __init__(self):

        self._cv_bridge = CvBridge()
# Subscriber
        self._sub = rospy.Subscriber('usb_cam/image_raw', Image, self.callback, queue_size=1)
# Publisher
        self._pub = rospy.Publisher('face', Image, queue_size=1)


    def callback(self, image_msg):
# Change of format
        cv_image = self._cv_bridge.imgmsg_to_cv2(image_msg, "bgr8")
# Cascade Classifier 
        face_cascade = cv2.CascadeClassifier('/faces/haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
          cv2.rectangle(cv_image,(x,y),(x+w,y+h),(0,0,0),2)
# Gray image
          roi_gray = gray[y:y+h, x:x+w]
# Colour image
          face = cv_image[y:y+h, x:x+w]
# Change of format
	  ros_image = self._cv_bridge.cv2_to_imgmsg(face, "bgr8")
	  self._pub.publish(ros_image)

    def main(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('detection')
    tensor = Detection()
tensor.main()


