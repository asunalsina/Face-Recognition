import cv2

# Save the frames of the video in a folder and returns the number of frames
def frame():
	cap = cv2.VideoCapture("prs1.mkv") # open the default camera

	flag, image = cap.read()
	count = 0

	while flag:
		flag,image = cap.read()
		cv2.imwrite("/frames/frame%d.jpg" % count, image)     # save frame as JPEG file
		if cv2.waitKey(10) == 27:                     # exit if Escape is hit
			break
		count += 1

	cap.release()
	cv2.destroyAllWindows()
	return count
