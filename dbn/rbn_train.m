function N=rbn_train(N,layer,X)

  npat=size(X,1);                      % number of patterns to train on
  mom=N.par.mom/npat;                   % Momentum term
  eps=N.par.eps*(1-N.par.mom)/npat;     % Fraction of the increment corresponding to 1-mom
  dec=N.par.dec*N.par.eps;              % Fraction of weight decay to match current eps

  % Positive phase: bottom-up activation on signal X0
  Y=layer_activate(N.W{layer},N.B{layer},X); % Sigmoid unit activations, given input signal X 
  Ybin=Y>rand(size(Y));                % Calculate binary probabilistic activations, to be used in the negative phase
  
  poscor=X'*Y;                        	% Input-Unit Correlations in the positive phase
  posgB=sum(X);                        	% +corr genBias
  posB=sum(Y);                          % +corr unitBias
  
  % Negative phase (top-down and bottom-up activations) 
  X1=layer_activate(N.W{layer}',N.gB{layer},Ybin);% Generative (input) activation given the unit activity from the positive fase  
  Y1=layer_activate(N.W{layer},N.B{layer},X1);  % Activate the layer on the data  
  
  negcor=X1'*Y1;                        % Input-Unit Correlations in the negative phase
  neggB=sum(X1);                        % -corr genBias
  negB=sum(Y1);                         % -corr unitBias

  % Weight Increments (momentum + delta-Corr - weight-decay)
  N.dW {layer} = mom * N.dW {layer} + eps * (poscor-negcor) - dec * N.W {layer};
  N.dgB{layer} = mom * N.dgB{layer} + eps * ( posgB- neggB) - dec * N.gB{layer};
  N.dB{layer} = mom * N.dB{layer} + eps * ( posB- negB) - dec * N.B{layer};
  
  % Update weights with the increments
  N.W{layer}  = N.W{layer} + N.dW{layer};  
  N.gB{layer}  = N.gB{layer} + N.dgB{layer};  
  N.B{layer}  = N.B{layer} + N.dB{layer};
end