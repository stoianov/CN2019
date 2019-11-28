%% Model training (2), MLP
% Make visible to matlab the directories with the DBN and MLP networks (our two libraries)
bp=pwd; addpath(bp,[bp filesep 'dbn'],[bp filesep 'digitnn']);
load('digit_model.mat','M');

% Init a MLP classifier
M.dbnlayer=3;                   % The dbn layer to be used as an input
ninp=M.DBN.lsize(M.dbnlayer);   % N of inputs = size of top DBN layer
nhid=50;                        % 50 MLP latent units
nout=10;                        % Number of target labels
nses=30;                        % 30 training sessions
nepochs=1000;                   % How many epochs within session
MLP=mlp_init([ninp nhid nout]); % Initialize a MLP
ER=[];                          % Storage for error

% Data
[X,Y,N,A]=digitdata(100);       % Get a subset of 100x10 test cases
ALL=dbn_activate(M.DBN,X);      % Activate the dbn-model on this data
for i=1:nses
  fprintf('\nIteration %d ..\n',i);
  % TRAINING
  MLP=mlp_train(MLP,ALL{M.dbnlayer},Y,nepochs); % Train the network on this data
  % TEST on random subset .. and training set for next sesion
  [X,Y,N,A]=digitdata(100);     % New 100x10 test cases with inclination up to 20 degrees
  ALL=dbn_activate(M.DBN,X);    % Activate the dbn
  R=mlp_activate(MLP,ALL{M.dbnlayer}); % Network response on this new data
  ER=digitnn_analysis(ER,R,N,A);% Visualize the error
end

M.MLP=MLP;                      % Add the MLP classifier to the model
save('digit_model.mat','M');    % Store the model for further analysis