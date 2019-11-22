function contrast_profile(NN)   % profile the effect of stimulus contrast 
[X,~,N,~]=digitdata(500);       % Load a subset with 500 items per digit 

%% Image contrast
SG=std(X,[],2);                 % Use the st.dev. of each image as a proxy for contrast
SG20=round(SG*20);              % Make a discrete scale by multiplying the image stdev by 20
SG20scale=min(SG20):max(SG20);  % The scale with all levels of the discrete SG20 property
SGscale=SG20scale/20;           % The scale in original units
nlev=numel(SG20scale);          % How many levels in this scale
ER=zeros(nlev,1);               % Storage for the average error for each contrast level

%% Activate the NNet and collect the error as a function of contrast
R=mlp_activate(NN,X);      % The network response on the data
[~,Rn]=max(R,[],2);             % Response Category (1..10) 
nerr=(Rn-1)~=N;                 % Network error on each pattern
for lev=1:nlev 
  ER(lev)=mean(nerr(SG20==SG20scale(lev))); % Average response accuracy
end

figure(6);                     
 plot(SGscale,ER);            % Show the response as a function of constrast
 title('Effect of image contrast (std)');
 xlabel('Contrast (measured as std)');
 ylabel('Err');
end