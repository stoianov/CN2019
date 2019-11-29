% Sensory (bottom-up) driven activation of a DBN on a set of data patterns X

function A=dbn_activate(N,X)

A=cell(N.nlayers,1);          	% Container for input data and activity of each layer
A{1}=X;                         % Keep the data as 1st (input) layer
for l=2:N.nlayers 		% The computational layers start from the 2nd layer
  A{l}=layer_activate(N.W{l},N.B{l},A{l-1}); % Activate layer l with input from the preceding layer
end

end
