#!/usr/bin/env python

import numpy as np
import os,glob,cv2
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge


class Save():
    def __init__(self):

        self._cv_bridge = CvBridge()
# Subscriber
        self._sub = rospy.Subscriber('face', Image, self.callback, queue_size=1)

    def callback(self, ros_image):
        
	cv_image = self._cv_bridge.imgmsg_to_cv2(ros_image, "bgr8")

	cv2.imshow("face", cv_image)

	answer = raw_input('Do you want to save the image? (yes/no): ')
# Loop to save the image or to ask for a valid input	
	while answer !='no' and answer != 'yes':
		answer = raw_input('Wrong answer, try again (yes/no): ')

	else:
		if answer == 'yes':
			n = raw_input('Name of the picture: ')
			print "Saving..."
			cv2.imwrite('/images/%s.jpg' %n, cv_image)

		else:
			print "No image will be saved"

	cv2.waitKey(1000)


    def main(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('save')
    tensor = Save()
tensor.main()
