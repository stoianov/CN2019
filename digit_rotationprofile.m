function digit_rotationprofile(M)    % plot the effect of rotation on response accuracy 
nlayers=M.DBN.nlayers;          % number of perception layers on each of which is fit a classifier  
[X,~,N,A]=digitdata(500);       % Load a subset with 500 items per digit 
Z=dbn_activate(M.DBN,X);        % Activate the dbn perception network on the data

% ROTATION SCALE
A10=round(A/10);                % -50 .. 50 degrees -> -5 .. 5  Turn the rotation to a scale with few levels
amin=min(A10);                  % highest level in the new scale
amax=max(A10);                  % lowest level in the new scale
range10=amin:amax;              % rotation scale in the new category units
range=range10*10;               % rotation scale in the original degrees (-50 .. 50) 
na=amax-amin+1;                 % Number of new rotation levels
ER=zeros(na,nlayers);           % Storage for average response for each rotation-level

% Collect the error for each classifier
for layer=1:nlayers
 R=mlp_activate(M.MLP{layer},Z{layer}); % The network response on the data
 [~,Rn]=max(R,[],2);            % Response Category (1..10) &
 E=(Rn-1)~=N;                   % Response accuracy per each pattern

 % Build the contrast profile for this classifier
 for i=1:na
   ER(i,layer)=mean(E(A10==range10(i))); % Average response error for each rotation level
 end
end

h_fg=figure(7); clf reset;      % Set figure properies (position, size, background, .. ) ready to save as image	
set(h_fg,'Position',[500,5000,500,400],'Renderer','zbuffer','Color',[1 1 1],'PaperPositionMode', 'auto');
 plot(range,ER);                % Show the response as a function of rotation
 title('Rotation profile');
 xlabel('Rotation (degrees)');
 ylabel('Classification error');
 legend({'MLP(Input)','MLP(DBN-Layer1)','MLP(DBN-Layer2)'});
print(h_fg,'-dpng','-painters','-r100','dig_dbn-mlp - rotation profile.png'); % Save the response profile as an image "response.png"
end
