# Recognition of digit images

This repository contains a pattern-recognition application of the back prop learning algorithm. 

*Content*:

- Data

  *Digits20x20.mat*

  ​	Dataset containing in total 5000 images of digits 0, 1, .. 9 in matlab format (500 images per class). The images are linearized as vectors. Each row in D.IMG contains one digit image.

  *digitdata_demo.m*

  ​	shows few sample images and their classes. 

  *digitdata.m*

  ​	Read the dataset and return training data.  Sample use:

  ​	[X, Y, N, A] = *digitdata*(100);

  ​    X:  a matrix of 100 randomly selected linearized images per each of the 10 categories (0,1,..9) 

     N: a vector with the class (digit) of each image

     A: rotation angle of each image

     Y: one-hot coding of each class

  

- Learning digit recognition with a MLP and online analysis

  ***digitnn***.m

  ​	specified params and trains a MLP network to recognize the images, using the back prop algorithm.
  
  ​    Learning is divided in sessions (e.g., n=10 sessions). 
  
  ​    Each session draws training data; trains the network on it, and analyzes the response on a new set of data that in turn becomes new training set.  
  
  *digitnn_analysis.m*
  
  ​	function that analyzes the response (overall error, confusion matrix ecc)	



To run the code, put all files in a single directory and run ***digitnn***