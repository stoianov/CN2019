function contrast_profile2(NN)   % profile the effect of stimulus contrast 
[X,~,N,~]=digitdata(500);       % Load a subset with 500 items per digit 

%% Image contrast
SG=std(X,[],2);                 % Use the st.dev. of each image as a proxy for contrast
SG20=round(SG*20);              % Make a discrete scale by multiplying the image stdev by 20
SG20scale=min(SG20):max(SG20);  % The scale with all levels of the discrete SG20 property
SGscale=SG20scale/20;           % The scale in original units
nlev=numel(SG20scale);          % How many levels in this scale
ER=zeros(nlev,1);               % Storage for the average error for each contrast level
Iplot=zeros(nlev,1);            % Index of examples images to be displayed for each contrast level 

%% Activate the NNet and collect the error
R=mlp_activate(NN,X);           % The network response on the data
[~,Rn]=max(R,[],2);             % For each pattern, response Category (1..10) is the category with strongest response 
nerr=(Rn-1)~=N;                 % Network error for each pattern: There is an error when the preferred digit is different from the real digit

%% Compute the contrast profile: the average error as a function of each level of categorized contrast
for lev=1:nlev
  I=find(SG20==SG20scale(lev)); % Index of all stimuli whose contrast-category is equal to SG20scale(lev)
  ER(lev)=mean(nerr(I));        % Average response accuracy of these stimuli
  Iplot(lev)=I(randi(numel(I)));% Keep the index of a randomly selected sample with this contrast level
end  

%% VISUALIZATION
figure(61);                     
 plot(SGscale,ER);              % Show the response as a function of constrast
 title('Effect of image contrast (std)');
 xlabel('Contrast (measured as std)'); 
 ylabel('Err');
 
figure(62); 
 for lev=1:nlev                 % Show the sample images whose indexes we stored 
   subplot(1,nlev,lev);         
   IMG=reshape(X(Iplot(lev),:),20,20); % Reshape the vectorized images back to 2D array 
   imagesc(IMG);                % Show the images
   axis off;                    % Take out axes which are not needed here
   title(sprintf('Contrast %.2f',SGscale(lev))); % Image title informs about the condition
   colormap gray;               % Grayscale images are more informative about contrast than color images
 end 
end