# **Sensor Data Filtering and Augmentation Tools**

## **Motivation**

This set of libraries contains various methods to help with sensor data
filtering, data augmentation, and data preparation for use in Keras and 
TensorFlow models along with computing and then plotting t-SNE, PCA, and ICA calculations.
Some functions from this library rely on a [data augmentation 
library](https://github.com/maddyarmstrong/Data-Augmentation-For-Wearable-Sensor-Data/blob/master/DataAugmentation.py)
found in another repository.  The filtering technique in this library hinges on a
multi-resoultion windowing approach coupled with variance filtering. This
technique was orginially developed to determine the start and end time of an
activity in a given window where the activity. The library has the capabilty to create different window types for data preprocessing, create training, validation, and testing sets from given percentages, and create tf.data.Dataset from inputted data. 

## Tutorial

A tutorial for how to use the libraries and their capabilites can be found **here**. The tutorial explores the how to use these libraries on an academic dataset of simulated smart watch data from [this study](https://archive.ics.uci.edu/ml/datasets/Dataset+for+ADL+Recognition+with+Wrist-worn+Accelerometer).

## Dependencies

 Most of these dependencies can be installed using ```pip``` excpet ```DataAugmentation.py``` which can be found [here](https://github.com/maddyarmstrong/Data-Augmentation-For-Wearable-Sensor-Data/blob/master/DataAugmentation.py). 
*  Python (3.8.5)
* [DataAugmentation.py](https://github.com/maddyarmstrong/Data-Augmentation-For-Wearable-Sensor-Data/blob/master/DataAugmentation.py)
* numpy (1.18.5)
* matplotlib (3.2.2)
* TensorFlow (2.3.0)
* seaborn (0.10.1)
* [transforms3d](https://matthew-brett.github.io/transforms3d/index.html)
* scikit-learn (0.23.1)
    * TSNE
    * PCA
    * FastICA
* ```from skimage.util.shape import view_as_windows```(0.17.2)
* ```from dataclasses import dataclass```
