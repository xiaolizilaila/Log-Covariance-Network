# Log-Covariance-Network
skeletal action-recognition

## Overview

This repository is trying to follow the paper "When Kernel Methods meet Feature Learning: Log-Covariance Network for Action Recognition from Skeletal Data" by Jacopo Cavazza, et al. In this paper, the authors propose an approach where a shallow
network is fed with a covariance representation of skeletal data.

Notice: This code is still under improvement. 

## Running the code

### Setup

Firstly, clone this repository using

`$ git clone https://github.com/xiaolizilaila/Log-Covariance-Network.git`

### Download the dataset

Here I tested on MSR-Aciton-3D dataset, and it can be found in http://research.microsoft.com/en-us/um/people/zliu/actionrecorsrc/

### Sample code

Run the example code using

`$ python LogCOV.py`
 
to get the covariance representation of skeletal data.
 
Then, run the next code using
 
`$ python full_connected_feed.py`
 
to train a single fully connected layer, the code is a modified version of tensorflow examples.
 
 ### Evaluate
 
 I used libsvm tools for classification on the features obtained from the single fully connected layer, the tools can be downloaded in https://www.csie.ntu.edu.tw/~cjlin/libsvm/
 
 ### Results
 
 The testing result is 90.06% while the paper achieved 97.4% on MSR-Aciton-3D dataset. In my opinion, the network may overfit caused by the small amount data if MSR-Aciton-3D dataset.
 
 ### Questions 
 
 Please feel free to pull requests for this code. For any questions, you can contact limengjietju@gmail.com.
 
