# **Traffic Sign Recognition** 


This document is the writeup for an exercise of an online course.

Using the data of an older competition to detect german traffic signs a neuronal network should be designed, implemented and evaluated.

The site of the original challenge is http://benchmark.ini.rub.de/?section=gtsrb

---

[//]: # (Image References)

[image1]: ./docu-images/images_of_all_classes.png "Samples of all classes"
[image2]: ./docu-images/histogram.png "Histogram"
[image3]: ./docu-images/random_samples_original.png "Random samples"
[image4]: ./docu-images/preprocessed_samples.png "Preprocessed samples"
[image5]: ./docu-images/augmented_samples.png "Augmented samples"
[image6]: ./docu-images/accuracy_plot.png "Accuracy plot"
[image7]: ./docu-images/new_images.png "New images"
[image8]: ./docu-images/new_images_preprocessed.png "New images preprocessed"
[image9]: ./docu-images/featuremap_5.png "Feature map 1"
[image10]: ./docu-images/featuremap_6.png "Feature map 2"
[image11]: ./docu-images/featuremap_layer2.png "Features layer 2"


# Summary

The following steps were done:

* Data Set Analysis: calulating some simple statistics of the data set with Python, NumPy and Pandas
* Exploratory Visualization: plotting some random samples of the images, plotting one sample of every type of traffic sign, plotting a histogram of the data
* Preprocessing: the images are preprocessed using OpenCV
* Model Architecture: After a first try with the original LeNet and an improved version with dropout regularization and more layers, an architecture from a paper which was created during and after the original challenge was reprogrammed in Tensorflow and used for further analysis [1].
* Model Training: the model was trained using Adam optimizer with cross entropy as loss function
* Solution Approach: Two stage convolutional neural network
* Aquiring new images and test the model
* Visualization of the features detected by the two network layer

# Data Set Analysis

## Basic statistics

The dataset with the traffic signs was split in

* 34799 samples in the train set
* 4410 samples in the validation set
* 12630 samples in the test set

The images have a size of 32x32 with 3 colors.

There are 43 classes of traffic signs in the data.

## Data exploration

First from all types of traffic signs contained in the data an image is plotted to have a first visual overview.

![Samples of all classes][image1]

* All the images have the same size and are cropped to only contain one traffic sign.
* There are traffic signs with different shapes and colors.
* The contrast of the images is poor on some samples and some samples are very noisy.

### Histogram

![Histogram of the traffic sign classes][image2]

The classes are not equally distributed. Of some classes there are about 2000 samples in the train set and for others only about 200 samples are contained.


## Data preprocessing

The images contained in the data are obtained from videos and are not preprocessed. The quality is often low like in a real world use case.

To help the neural network learning the traffic sign classes it's needed to improve and normalize the data as good as possible.

The first try was just normalizing the pixels images to the range of [0.0, 1.0]. This is necessary for processing the data in many neural networks depending on the activation function.

With this simple normalization the accuracy was not as good as needed since the contrast is poor on many examples.

Next try was with contrast normalization using OpenCV functions. This gives a high improvement and even for humans the signs are better to recognize.

But in some images there are already very bright regions (sky) and dark regions (traffic sign in a shadow) so the performance was not good enough. So local contrast adjustment was used at the end using OpenCV (Contrast Limited Adaptive Histogram Equalization [2]).

### Preprocessing shown in examples

Some sample images before the preprocessing step.

![Traffic signs before preprocessing][image3]

And after preprocessing:

![Traffic signs after preprocessing][image4]

## Image augmentation

Since the dataset is relatively small the neural network has problem to generalize the traffic sign classes. To help to generalize better and to prevent overfitting the images are ramdomly augmentated while training.

For augmentation the following transformation were done on the original images:

* Conversion to grayscale
* Rotation in the range of [-15°, +15°]
* Translation in the range of [-5, +5] pixels
* Zoom in the range of [0.9, 1.1]

Tests with addition of random noise gave worse result.

Conversion to grayscale helped to reduce the training time and in [1] it was found out that it can even increase the performance of the network.
The other augmentation techniques are successfully applied in many machine learning tasks and are state of the art. All ranges are checked with sample images to give good results, the same or similar values are used in [1].

### Example augmentation

![Traffic signs after augmentation][image5]

# Neural network architecture

From [Sermanet, LecCun, 2011] it is known that an accuracy of more than 99% percent is possible. First tests with the simple LeNet architecture with Dropout for regularization gave much worse results.

Improved versions with more and larger layers gave better results but were slow during training phase.

Since the paper described a succesfull architecture with a relatively small network it was tried to reproduce it using Tensorflow.

The network uses two stages with convolution and max pooling followed by two fully connected layers. To allow the fully connected layers to use low and high level features the output of stage two contains the the output of layer 2 merged with the output of layer 1 (with an additional max pooling).

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x108 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x64 				|
| Dropout               | keep probability 0.65                          |
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x200  |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x200   				|
| Dropout               | keep probability 0.65                          |
| Max pooling of layer 1| 2x2 stride,  outputs 7x7x108                  |
| Flatten               | flatten of layer 2 and layer 1 with max pooling, outputs 10292x1|
| Fully connected		| 100 nodes         							|
| Dropout               | keep probability 0.65                          |
| Fully connected		| 42 nodes         						    	|
| Softmax				|           									|

## Model Optimization

For image classification convolutional neural networks are the state of the art. Most classes of objects in images can be detected finding low level features which can be combinded to higher and higher level features. Same features can be at different positions of the image so convolution helps reducing the size of the network and share parameters.

Since larger neural networks have the tendency to overfit the training data Dropout is used for regularization. It forces the follwing layers not to rely on a small set of features to classify objects which migh not be available in new images.

The size of the network was taken from [1] where it gave the best results.

While running the model training a plot of the train and validation accuracy was done. Using the plot the parameters were optimized to have neither an over- nor underfitted model. For the best result the training accuracy was 1.0 and the validation accuracy was 0.991. Trying to increade droput rate or image augmentation gave for both accuracies worse results.

![Accuracy plot][image6]

Since the model was interrupted the total number of epochs of the x-axis is the correct number.

## Neural network training

The neural network was trained using the Adam optimizer with cross entropy as loss function. Learning rate started with 0.0005 with an exponential decay, batchsize was 128. For image augmentation rotation, shift and zoom was used. Only one transformation was done since this gave the best result.

The last training had the best model after 129 epochs.

## Results

The final model results were:
* training set accuracy of 1.0
* validation set accuracy of 0.991
* test set accuracy of 0.979

# Test model on new images

The following additional traffic signs were used for testing the model

![New images][image7]

After preprocessing:

![New images after preprocessing][image8]

Since the new images are of relatively high quality there were no problems for the model to classify correctly.

Here are the results of the prediction:

'Keep right' 'No passing' 'Road work' 'Speed limit (30km/h)' 'No entry'
 'No vehicles' 'No vehicles' 'Traffic signals' 'No vehicles'
 'Turn right ahead' 'Speed limit (30km/h)'

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Keep right      		| Keep right   									| 
| Road work     			| Road work 										|
| Speed limit (30km/h)					| Speed limit (30km/h)											|
| No entry	      		| No entry					 				|
| No vehicles			| No vehicles      							|
| No vehicles			| No vehicles      							|
| Traffic signals			| Traffic signals      							|
| No vehicles			| No vehicles      							|
| Turn right ahead			| Turn right ahead      							|
| Speed limit (30km/h)			| Speed limit (30km/h)      							|

The accuracy is 1.0 which is comparable to the test set accuracy of 0.98.

For all traffic signs the probability for the predicted class is 1.0. Even for the worst image the second highest probability is 0.000015.

# Visualizing the Neural Network

The features detected by the first two layers are shown in the next images.

First there are two visualizations of the features detected by the first layer of the network with first an triangular sign and second with a round sign.

![New images][image9]

![New images][image10]

The shapes of the sign can be clearly seen in the two images.

The next image shows the output of the second layer which weight the different features from the first layer.

![New images][image11]

The result can not be interpreted as easy as the output from the first layer.

# References

[1](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) Pierre Sermanet,Yann LeCun: "Traffic Sign Recognition with Multi-Scale Convolutional Networks", 2011 

[2](https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html) Tutorial py histogram equalization
