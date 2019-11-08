% N: MLP structure
% X: Input data (the features of each pattern)
% T: Target data
function [N,e]=mlp_backprop2(N,X,T)

 lc=0.01;                      % Learning coefficient
 mc=0.3;                        % momentum coefficient
 
 [Y, H]=mlp_activate(N,X);      % Activate the MLP on data
 
 E=T-Y;                         % Errors for each data patterns
 dY = E.*Y.*(1-Y);              % Deltas for the output
 dH = (dY*N.ow').*H.*(1-H);     % Deltas for the hidden layer 
 
 N.dow= mc * N.dow + (1-mc)*lc * (H'*dY);
 N.dob= mc * N.dob + (1-mc)*lc * sum(dY);
 N.dhw= mc * N.dhw + (1-mc)*lc * (X'*dH);
 N.dhb= mc * N.dhb + (1-mc)*lc * sum(dH);
 
 N.ow = N.ow + N.dow;           % ow: [nh,no]
 N.ob = N.ob + N.dob;           % ob  [1 ,no]
 N.hw = N.hw + N.dhw;           % hw  [in,nh]
 N.hb = N.hb + N.dhb;           % hb  [1 ,nh]
 
 e=mean(abs(E(:)));             % Average error over all output units and all data
 
end