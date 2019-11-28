% Create a DBN with random weights
% Input: vector "lsize" contains the size of the sensory data and the layers 
% e.g. lsize=[100 50 50] indicates a two-layer DBN with 100 sensory (data) units and two layers of 50 units each

function N=dbn_create(lsize)
nlayers=numel(lsize);		% computational layers start from the 2nd layer

N.nlayers=nlayers;              % the number of the layers
N.lsize=lsize;                  % keep the network size
N.W=cell(nlayers,1);            % empty cell array for weights
N.gB=cell(nlayers,1);           % empty cell array for generative biases
N.B=cell(nlayers,1);           	% empty cell array for biases
s=0.1;                          % st.dev. of the gaussian used to initialize the weights

for l=2:N.nlayers               % Init all weights and biases, starting from the 2nd layer
  ninputs=lsize(l-1);           % the size of the input of this layer
  nunits=lsize(l);              % the number of hidden units in this layer
  % Weights
  N.W{l}=randn(ninputs,nunits,'single')*s; 	% random weights: single-precision gaussian randoms
  N.gB{l}=randn(1      ,ninputs,'single')*s; 	% .. biases for each input (for the generative phase)
  N.B{l}=randn(1      ,nunits,'single')*s; 	% .. biases for each unit (for bottom-up unit activations)
  % Learning increments
  N.dW{l}=zeros(ninputs,nunits,'single');
  N.dgB{l}=zeros(1,ninputs,'single');
  N.dB{l}=zeros(1,nunits,'single');
end

end
