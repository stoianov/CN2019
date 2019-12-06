%% Hybrid deep DBN-MLP perception model
%  We assume that the unsupervised DBN network is already trained

% Make visible to matlab the directories with the DBN and MLP networks (our two libraries)
bp=pwd; addpath(bp,[bp filesep 'mlp'],[bp filesep 'dbn'],[bp filesep 'digitnn']);

fname='digitmodel_dbn(100_100)' % The name of the mat file with trained DBN perception network
load(fname,'M');                % Load this file
nlayers=M.DBN.nlayers;          % Number of processing layers (including the 1st input layer)
Leg={};                         % Legend for the error plots
nses=30;                        % Training epochs per session
nepochs=1000;                   % Epochs per session
lcoef=0.002;                    % Learning coefficient
noisesd=0.0;                    % std of perceptual noise
ERR=nan(nlayers,nses);          % Empty storage matrix for Session-error

for layer=1:3                   % Train classifiers on each level of representation
                                % !! Layer-1 is the input !!
 fprintf('\nTrain a classifier on representat layer %d: ',layer);
 % Init the classifier
 ninp=M.DBN.lsize(layer);       % N of inputs = size of the selected layer
 MLP=mlp_init([ninp 50 10]);    % Initialize a MLP with 50 hidden and 10 output units

 % Data
 [X,Y,N,A]=digitdata(100);      % Get a subset of 100x10 test cases
 noise=randn(size(X))*noisesd;
 Z=dbn_activate(M.DBN,X+noise); % Activate the dbn-model on this data
 
 for i=1:nses

  MLP=mlp_train(MLP,Z{layer},Y,nepochs,lcoef); % Train the network on this data
  
  % New Data and TEST
  [X,Y,N,A]=digitdata(100);     % New 100x10 test cases with inclination up to 20 degrees
  noise=randn(size(X))*noisesd; 
  Z=dbn_activate(M.DBN,X+noise);% Activate the dbn
  R=mlp_activate(MLP,Z{layer}); % Network response on this new data
  
  % Error analysis and plot
  [CM,err]=digitnn_confmatrix(R,N);
  ERR(layer,i)=err;            % Add the current session error to the error trend
  Leg{layer}=sprintf('Layer %d Err=%.3f',layer,err); % Legend of error trend
  digitnn_errorplot(CM,ERR,Leg,layer,2);% Plot the confusion matrix and the error trend
  fprintf('.');        
  
 end
 
 M.MLP{layer}=MLP;              % Store the MLP classifier
 M.ERR=ERR;                     % Store the learning trend (for all layers)
 M.CM{layer}=CM;                % Store the confusion matrixes
 M.Leg=Leg;                     % Store the legend with brief error description
end

save(sprintf('%s mlp(noise%.2f lc%.3f).mat',fname,noisesd,lcoef),'M'); % Store the model for further analysis