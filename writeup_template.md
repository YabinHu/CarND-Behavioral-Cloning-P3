# **Behavioral Cloning**

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24).

The first layer is a normalization layer, transforming value range of input data from [0.0, 255.0] to [-1.0, +1.0].

The second layer is a convolution layer with 3 1X1 filters. Employing this has the effect of transforming the color space of the images, which is suggested in [this article](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.6e5qnvj9o) by *Vivek Yadav*. Using 3 1X1 filters allows the model to choose its best color space.

The color space transformation layer is followed by 2 convolutional blocks each comprised of 24 and 36 filters of size 5X5. These convolution layers were followed by 3 additional convolutional blocks each comprised of 64 filters of size 3X3. All the convolution layers use the 'same' padding.

After a flatten layer, 3 fully connected blocks are introduced. All the FC layers have a BatchNormalization layer before it and a Dropout layer after.

All the layers have exponential relu (ELU) as activation function except the last one, which is a one-output layer with linear activation (to handle this regression problem).

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 108 and 111).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 13, test_size = 0.25). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 115).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section.

### Solution Design

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the vgg16 model. I thought this model might be appropriate because it was powerful enough to work really well on the ImageNet datasets. In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error both on the training set and on the validation set. However, the learned model performed badly during simulation, out of my expect, no mention that the training process is very long.

I then turn to the Nvidia model, because I thought it is simpler and maybe good enough for this project according to their paper. I tried to modify a bit on the model. Though the mse error turned to be very low on both training and validation data, my model still performed badly on the very early steps of the track.

After digging a little bit on the forum, I realized the key to this project is the way we preprocess and augment the data. I then recorded my own data and mixed them with Udacity official data, hoping bigger datasets make better performance. But this didn't work either. At last I referred to the augmentation technology proposed by *Kaspar Sakmann* in [this post](https://medium.com/@ksakmann/behavioral-cloning-make-a-car-drive-like-yourself-dc6021152713#.9u3z7zgzn).

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 80-113) is as follows:
```
         OPERATION           DATA DIMENSIONS   WEIGHTS(N)   WEIGHTS(%)

             Input   #####     64   64    3
            Lambda   ????? -------------------         0     0.0%
                     #####     64   64    3
            Conv2D   ????? -------------------        12     0.0%
                     #####     64   64    3
     ZeroPadding2D   \|||/ -------------------         0     0.0%
                     #####     68   68    3
            Conv2D   ????? -------------------      1824     0.5%
               elu   #####     32   32   24
     ZeroPadding2D   \|||/ -------------------         0     0.0%
                     #####     36   36   24
            Conv2D   ????? -------------------     21636     6.2%
               elu   #####     16   16   36
      MaxPooling2D   YYYYY -------------------         0     0.0%
                     #####      8    8   36
     ZeroPadding2D   \|||/ -------------------         0     0.0%
                     #####     10   10   36
            Conv2D   ????? -------------------     20800     6.0%
               elu   #####      8    8   64
     ZeroPadding2D   \|||/ -------------------         0     0.0%
                     #####     10   10   64
            Conv2D   ????? -------------------     36928    10.6%
               elu   #####      8    8   64
     ZeroPadding2D   \|||/ -------------------         0     0.0%
                     #####     10   10   64
            Conv2D   ????? -------------------     36928    10.6%
               elu   #####      8    8   64
      MaxPooling2D   YYYYY -------------------         0     0.0%
                     #####      4    4   64
     ZeroPadding2D   \|||/ -------------------         0     0.0%
                     #####      6    6   64
            Conv2D   ????? -------------------     18464     5.3%
               elu   #####      4    4   32
     ZeroPadding2D   \|||/ -------------------         0     0.0%
                     #####      6    6   32
            Conv2D   ????? -------------------      9248     2.7%
               elu   #####      4    4   32
      MaxPooling2D   YYYYY -------------------         0     0.0%
                     #####      2    2   32
           Flatten   ||||| -------------------         0     0.0%
                     #####         128
BatchNormalization    μ|σ  -------------------       512     0.1%
                     #####         128
             Dense   XXXXX -------------------     66048    19.0%
               elu   #####         512
           Dropout    | || -------------------         0     0.0%
                     #####         512
BatchNormalization    μ|σ  -------------------      2048     0.6%
                     #####         512
             Dense   XXXXX -------------------    131328    37.8%
               elu   #####         256
           Dropout    | || -------------------         0     0.0%
                     #####         256
BatchNormalization    μ|σ  -------------------      1024     0.3%
                     #####         256
             Dense   XXXXX -------------------       257     0.1%
                     #####           1
```

#### 3. Creation of the Training Set & Training Process

This final model was generated by using Udacity data only. In addition, employing several data augmentation technology introduced by *Kaspar Sakmann* (in BC_helper.py).

To augment the training data sat, the following steps were taken:
* randomly select the image from center or left or right camera
* random_shear the image
* random_crop the image and resize it to 64x64
* random_flip the image
* random_brightness the image


To augment the validation data sat, use random_crop function only.

I finally randomly shuffled the data set and put Y% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 20 as evidenced by smoothly passing several crash-points in the track. I used an adam optimizer so that manually training the learning rate wasn't necessary.
