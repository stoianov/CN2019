function digit_contrastprofile(M)   % profile the effect of stimulus contrast 
nlayers=M.DBN.nlayers;          % number of perception layers on each of which is fit a classifier  
[X,~,N,~]=digitdata(500);       % Load a subset with 500 items per digit 
Z=dbn_activate(M.DBN,X);        % Activate the dbn perception network on the data

% Image contrast
SG=std(X,[],2);                 % Use the st.dev. of each image as a proxy for contrast
SG20=round(SG*20);              % Make a discrete scale by multiplying the image stdev by 20
SG20scale=min(SG20):max(SG20);  % The scale with all levels of the discrete SG20 property
SGscale=SG20scale/20;           % The scale in original units
nlev=numel(SG20scale);          % How many levels in this scale
ER=zeros(nlev,nlayers);         % Storage for the average error for each contrast level

% Collect the error for each classifier
for layer=1:nlayers
 R=mlp_activate(M.MLP{layer},Z{layer}); % The network response on the data
 [~,Rn]=max(R,[],2);            % For each pattern, response Category (1..10) is the category with strongest response 
 E=(Rn-1)~=N;                   % Network error for each pattern: There is an error when the preferred digit is different from the real digit

 % Compute the contrast profile: the average error as a function of each level of categorized contrast
 for lev=1:nlev
  I=find(SG20==SG20scale(lev)); % Index of all stimuli whose contrast-category is equal to SG20scale(lev)
  ER(lev,layer)=mean(E(I));     % Average response accuracy of these stimuli
 end  
end

figure(6);                     
 plot(SGscale,ER);              % Show the response as a function of constrast
 title('Contrast profile');
 xlabel('Contrast (measured as image std)'); 
 ylabel('Classification error');
 legend({'MLP(Input)','MLP(DBN-Layer1)','MLP(DBN-Layer2)'});
end