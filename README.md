# Face Recognition

Convolutional neural network for face recognition using TensorFlow and the algorithm for face detection created by Paul Viola and Michael Jones.

## Face folder

This folder contains the code to create a ROS package to run the system in a mobile robot. In the launch folder there is a .launch file to call all the necessary nodes at the same time. The neural network needs to be trained before using this package since the model file is not in the folder.

#### How to use

1. Create a [workspace](http://wiki.ros.org/catkin/Tutorials/create_a_workspace)
2. Copy in the src folder the face package
3. Run again catkin_make

If there is a problem with OpenCV while running catkin_make, it will be necessary to locate OpenCVConfic.cmake or opencv_config.cmake in your computer and paste the path in the CMakeLists.txt file that is inside the face-pkg folder. You have to copy the following line before the find_package() line:

**set(OpenCV_DIR /your_path)**

Then run again catkin_make. Once catkin_make runs without errors the package will be installed and ready to use.

## Neural network folder

Contains the code to create the dataset and train the neural network. 
