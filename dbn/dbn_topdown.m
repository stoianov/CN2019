% Generative (top-down) activation of a DBN, starting from upper-layer activities Y
% Note that we use the transposed weight matrix and input biases iB

function A=dbn_generate(N,Y)

A=cell(N.nlayers,1);          	% Container for the data and the activity of each layer
A{end}=Y;                       % Top-layer activity
for l=N.nlayers:-1:2
  A{l-1}=layer_activate(N.W{l}',N.gB{l},A{l}); % Activate layer l given the activity of the superior level
end

end
