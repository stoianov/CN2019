function N=mlp_init(sz)

N.sz=sz;                  		% network size (we expect 3 layers)
N.nlayers = numel(sz);        	% the total number of the layers (including inp and out)

% Weights and Biases of the 1st and 2nd layer
N.hw=randn(sz(1),sz(2));        % random weights of the hidden layer (n-inp x n-hid)
N.hb=zeros(1,sz(2));           	% null bias
N.ow=randn(sz(2),sz(3));        % random weights of the ouput layer (n-hid x n-out)
N.ob=zeros(1,sz(3));           	% null bias of output units

% Weight/Bias increments        Parameters used to speed-up the learning (avoid local minima)
N.dhw=zeros(size(N.hw));        
N.dhb=zeros(size(N.hb));
N.dow=zeros(size(N.ow));
N.dob=zeros(size(N.ob));

end