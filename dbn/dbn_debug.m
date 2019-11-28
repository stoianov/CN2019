% Test for error-free program execution

% Create a random network of size: 10 sensory units (L1), 3 units in L2, 4 units in L3
N=dbn_create([10 3 5]) 

% Activate the network on two random sensory patterns of size 10
A=dbn_activate(N,rand(2,10))

% Train the network on 1000 random patterns for 10 epochs 
N=dbn_train(N,rand(1000,10),10)

% Activate again the network on 2 random patterns
A=dbn_activate(N,rand(2,10))

% Apply a top-down generative pass on the activations of the last layer
A=dbn_topdown(N,A{end})