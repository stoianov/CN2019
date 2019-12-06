% N: MLP structure
% X: Input data (the features of each pattern)
% T: Target data (desired output activations for each input patterns)
% epochs: train on how many epochs
% lcoef: learning coefficient

function N = mlp_train(N,X,T,epochs,lcoef)
  N.ep=epochs;                  % Number of training steps
  N.lc=lcoef;                   % Learning coefficient
  N.err = zeros(epochs,1);      % Allocate space for errors

  for i= 1:epochs               % cycle
    [N,e]=mlp_backprop(N,X,T);	% A single optimization pass
    N.err(i)=e;                 % Keep errors in a vector
  end

end
