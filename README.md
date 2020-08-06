# **Sensore Data Filtering and Augmentation Tools**

## **Motivation**

This set of libraries contain various methods to help with sensor data
filtering, data augmentation, and data preperation of use in Keras and 
TensorFlow models along with running visualizations such as t-SNE, PCA, and ICA.
Some functions from this library rely on a [data augmentation 
library](https://github.com/maddyarmstrong/Data-Augmentation-For-Wearable-Sensor-Data/blob/master/DataAugmentation.py)
found in another repository.  The filtering technique in this library hinge on a
multi-resoultion windowing approach coupled with variance filtering. This
technique was  orginially developed to determine the start and end time of an
activiyt in a given window where the activity. These libraries also grew and
were  created while developing and running an amalgamation of experiments on
sensor data. Specifically, on the impact that using different windows of the 
data while training a model to recognize whether a given activity was taking
place in the window. 

## Tutorial

A tutorial for how to use the libraries and their capabilites can be found here. The tutorial explores the how to use these libraries on an academic dataset of simulated smart watch data from [this study](https://archive.ics.uci.edu/ml/datasets/Dataset+for+ADL+Recognition+with+Wrist-worn+Accelerometer).

## Dependencies
 
 Most of these dependencies can be installed using ```pip``` excpet ```DataAugmentation.py``` which can be found [here](https://github.com/maddyarmstrong/Data-Augmentation-For-Wearable-Sensor-Data/blob/master/DataAugmentation.py). 
* [DataAugmentation.py](https://github.com/maddyarmstrong/Data-Augmentation-For-Wearable-Sensor-Data/blob/master/DataAugmentation.py)
* numpy
* matplotlib
* TensorFlow
* seaborn
* sklearn
    * TSNE
    * PCA
    * FastICA
* ```from skimage.util.shape import view_as_windows```
