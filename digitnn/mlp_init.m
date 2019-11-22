% Creates a new 2-layer multilayer perceptron.
% Network size is given in the input parameter sz
% sz=[n-input units, n-hidden units, n-output units]
% Returns variable N: a structure containing the layer weights, ready to be trained

function N=mlp_init(sz)

N.sz=sz;                  		  % network size (we expect 3 layers)
N.nlayers = numel(sz);        	% the total number of the layers (including inp and out)

N.hw=randn(sz(1),sz(2));        % random weights of the hidden layer (n-inp x n-hid)
N.hb=zeros(1,sz(2));           	% null bias

N.ow=randn(sz(2),sz(3));        % random weights of the ouput layer (n-hid x n-out)
N.ob=zeros(1,sz(3));           	% null bias of output units

end
