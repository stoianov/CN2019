% Test the effect of noise level on several networks combined in a cell-array. 
% NN: the cell array with networks. LEG: concise description of each network 
% E.g.: noise_profile({NN,NN0},{'trained','baseline'})

function noise_profile(NN,LEG)  % plot the effect of noise on response accuracy of several networks 
nnet=numel(NN);                 % NN is a cell array; each element is one network.
[X,~,N,~]=digitdata(500);       % Load a subset with 500 items per digit 

sig=(0:0.05:0.5);               % Test noise levels (st.dev. of zero-centered noise)
nlev=numel(sig);                % How many levels
AC=zeros(nlev,nnet);            % Storage for average error for each noise level and each network

%% TEST the behavior for each noise level
for lev=1:nlev
 noise=randn(size(X))*sig(lev); % Generate gaussian noise
 Xn=max(0,min(1,X+noise));      % Add the noise to the input, and rectify [0-1]
 for net=1:nnet
   R=mlp_activate(NN{net},Xn);  % Network response on this new data
   [~,Rn]=max(R,[],2);          % Response Category (1..10) &
   ac=(Rn-1)==N;                % Response accuracy of each pattern
   AC(lev,net)=mean(ac);        % Average response accuracy
 end
end

figure(5);  
 plot(sig,AC);                  % Show the response as a function of noise level
 title('Effect of noise level');
 xlabel('Noise level (st.dev.)');
 ylabel('Accuracy');
 ylim([0 1]);
 legend(LEG);                   % Legend says which are the networks

end