%% PARAMS
nhid=50;                    % number of hidden units
nses=30;                    % number of training sessions
nepochs=1000;               % number of iterations per session

%% DATA
[X,Y,N,A]=digitdata(100);   % Select 100 cases per each digit class

%% INIT
ninp=size(X,2);             % Number of input units
nout=10;                    % Number of target labels
NN=mlp_init([ninp nhid nout]);
NN0=NN;                     % Keep the initial state of the network
ER=[];                      % Storage for error

for i=1:nses
  fprintf('\nIteration %d ..\n',i);
  %% TRAINING
  NN=mlp_train(NN,X,Y,nepochs); % Train the network on this data

  %% TEST on random subset .. and training set for next sesion
  [X,Y,N,A]=digitdata(100); % New 100x10 test cases with inclination up to 20 degrees
  R=mlp_activate(NN,X);     % Network response on this new data
  
  %% ANALYSIS
  ER=digitnn_analysis(ER,R,N,A);
end