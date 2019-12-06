% N: MLP structure
% X: Input data (the features of each pattern)
% T: Target data
function [N,e]=mlp_backprop(N,X,T)
 lc=N.lc;                       % Learning coefficient
 [Y, H]=mlp_activate(N,X);      % Activate the MLP on data 
 E=T-Y;                         % Errors for each data patterns
 
 dY = E.*Y.*(1-Y);              % Deltas at the output layer
 dW = (H'*dY);                  % Weight change 
 N.ow = N.ow + lc * dW;         % ow: [nh,no]
 N.ob = N.ob + lc * sum(dY);   % ob  [1 ,no]
 
 dH = (dY*N.ow').*H.*(1-H);     % Deltas at the hidden layer  
 dW = (X'*dH);                  % Weigh changes 
 N.hw = N.hw + lc * dW;         % hw  [in,nh]
 N.hb = N.hb + lc * sum(dH);   % hb  [1 ,nh]
 
 e=mean(abs(E(:)));             % Average error over all output units and all data

end
