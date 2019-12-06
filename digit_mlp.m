%% Model training (2), MLP-classifier

% Make visible to matlab the directories with the DBN and MLP networks (our two libraries)
bp=pwd; addpath(bp,[bp filesep 'mlp'],[bp filesep 'dbn'],[bp filesep 'digitnn']);
fname='digit_model.mat';        % File with stored DBN model
load(fname,'M');
layer=3;                        % Which representation layer to used as input for the classifier

% Init a MLP classifier
ninp=M.DBN.lsize(M.dbnlayer);   % N of inputs = size of top DBN layer
nhid=50;                        % 50 MLP latent units
nout=10;                        % Number of target labels
nses=30;                        % 30 training sessions
nepochs=1000;                   % How many epochs within session
lcoef=0.003;                    % Learning coefficient
MLP=mlp_init([ninp nhid nout]); % Initialize a MLP
ERR=[];                         % Error trend

% Data
[X,Y,N,A]=digitdata(100);       % Get a subset of 100x10 test cases
Z=dbn_activate(M.DBN,X);        % Activate the dbn-model on this data
fprintf('Training MLP on the 3rd layer:');
for i=1:nses
  % TRAINING
  MLP=mlp_train(MLP,Z{layer},Y,nepochs,lcoef); % Train the network on this data
  % TEST on random subset .. and training set for next sesion
  [X,Y,N,A]=digitdata(100);     % New 100x10 test cases with inclination up to 20 degrees
  Z=dbn_activate(M.DBN,X);      % Activate the dbn
  R=mlp_activate(MLP,Z{layer}); % Network response on this new data
  
  [CM,mERR]=digitnn_confmatrix(R,N);
  ERR(i)=mERR;                  % Error trend
  digitnn_errorplot(CM,ERR);    % Plot the confusion matrix and the error trend
  fprintf('.',i);
end

M.layer=layer;
M.MLP=MLP;                      % Add the MLP classifier to the model
M.ERR=ERR;
M.CM=CM;                        % Keep the error trend and the Confusion matrix
save(fname,'M');                % Store the model for further analysis