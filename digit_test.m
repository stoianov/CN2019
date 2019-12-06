bp=pwd; addpath(bp,[bp filesep 'mlp'],[bp filesep 'dbn'],[bp filesep 'digitnn']);

% Assumes a pre-trained DBN (unsupervised perceptual network) and MLP (digit classifier) 
load('digit_model.mat','M');

% Data: a subset of 300x10 images of handwritten digits
[X,Y,N,A]=digitdata(100);

% Activate the unsupervised visual perception network
Z=dbn_activate(M.DBN,X);

% Activate the classifier on the pre-processed data
R=mlp_activate(M.MLP,Z{M.dbnlayer}); 

% Analize the performance of the classifier
[CM,mERR]=digitnn_confmatrix(R,N);
