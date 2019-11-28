% Make visible to matlab the directories with the DBN and MLP networks (our two libraries)
bp=pwd; addpath(bp,[bp filesep 'dbn'],[bp filesep 'digitnn']);

% DATA
[X,Y,N,A]=digitdata(300);       % Select 200 cases per each digit class
ninp=size(X,2);

% PARAMS
dbnsize=[ninp 100 50];          % n=400 sensory-layer units, n=100 L2-units, n=100 L3-units
dbnepochs=500;                  % number of DBN-epochs

% Create and Train
DBN=dbn_create(dbnsize);        % initialize a DBN model
DBN=dbn_train(DBN,X,dbnepochs); % train the DBN on 100 epochs

% Show the receptive fields
dbn_plot(DBN,[20 20],10);   

M.DBN=DBN;                      % Store the sensory preprocessor as field of model "M"
save('digit_model.mat','M');    % Save the model