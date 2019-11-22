# Visual perception of handwritten digits

This repository contains a pattern-recognition application of the back prop learning algorithm and analysis of the network performance. 

*Content*:

- **Data**

  `Digits20x20.mat`

  ​	Dataset containing in total 5000 images of digits 0, 1, .. 9 in matlab format (500 images per digit). The images are linearized as vectors. Each row in D.IMG contains the image of one digit.

  `digitdata_demo.m`

  ​	shows few sample images and their classes. 

  `digitdata.m`

  ​	Read the dataset and return training data.  Sample use:

  ​	[X, Y, N, A] = *digitdata*(100);

     X:  a matrix of 100 randomly selected linearized images per each of the 10 categories (0,1,..9) 
     N: a vector with the class (digit) of each image
     A: rotation angle of each image
     Y: one-hot coding of each class


- **Learning** handwritten digit recognition with the MLP network and analysis

  `digitnn.m`

  specifies params and trains a MLP network to recognize the images, using the back prop algorithm.
  
  Learning is divided in sessions (e.g., n=10 sessions). 
  
  Each session draws a subset of training data; trains the network on it, and analyzes the response on a new subset that in turn becomes training data for the next training session.  
  
  `digitnn_analysis.m`
  
  A function that analyzes the overall response (overall error, confusion matrix ecc)
  
- **Analysis** of the simulation

 `contrast_profile.m`
 
 `noise_profile.m`
 
 `rotation_profile.m`
 
 `plot_weights.m`
