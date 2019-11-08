% N: MLP structure
% X: Input data (the features of each pattern)
% T: Target data
function [N,e]=mlp_backprop(N,X,T)

 lc=0.001;                      % Learning coefficient
 [Y, H]=mlp_activate(N,X);      % Activate the MLP on data
 
 E=T-Y;                         % Errors for each data patterns
 dY = E.*Y.*(1-Y);              % Deltas for the output
 dH = (dY*N.ow').*H.*(1-H);     % Deltas for the hidden layer  
 
 N.ow = N.ow + lc * (H'*dY);    % ow: [nh,no]
 N.ob = N.ob + lc * mean(dY);   % ob  [1 ,no]
 N.hw = N.hw + lc * (X'*dH);    % hw  [in,nh]
 N.hb = N.hb + lc * mean(dH);   % hb  [1 ,nh]
 
 e=mean(abs(E(:)));             % Average error over all output units and all data

end
