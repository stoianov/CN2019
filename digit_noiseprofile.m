% Test the effect of noise level on a deep perception model. 
% M: a hybrid model consisting of unsupervised perception and supervised classifier network 
% E.g.: digit_noiseprofile(M)

function digit_noiseprofile(M)   
nlayers=M.DBN.nlayers;          % number of perception layers on each of which is fit a classifier  
[X,~,N,~]=digitdata(500);       % Load a subset with 500 items per digit
sig=(0:0.03:0.3);               % Test noise levels (st.dev. of zero-centered noise)
nlev=numel(sig);                % How many noise levels
ER=zeros(nlev,nlayers);         % Storage for average error for each noise level and each network

%% TEST the behavior for each noise level
for lev=1:nlev
 noise=randn(size(X))*sig(lev); % Generate gaussian noise
 Xn=max(0,min(1,X+noise));      % Add the noise to the input, and rectify [0-1]
 Z=dbn_activate(M.DBN,Xn);      % Activate the dbn perception network
 
 for layer=1:nlayers            % Test the effect of noise on each classifier
   R=mlp_activate(M.MLP{layer},Z{layer}); % Activate the classifier
   [~,Rn]=max(R,[],2);          % Response Category (1..10) &
   ER(lev,layer)=mean((Rn-1)~=N);% Response accuracy of each pattern
 end
end

figure(5);  
 plot(sig,ER);                  % Show the response as a function of noise level
 title('Noise profile');
 xlabel('Level (st.dev.) of gaussian noise');
 ylabel('Classification error');
 legend({'MLP(Input)','MLP(DBN-Layer1)','MLP(DBN-Layer2)'});
end