function rotation_profile(NN)    % plot the effect of rotation on response accuracy 
%% TEST SUBSET
[X,~,N,A]=digitdata(500);       % Load a subset with 500 items per digit 
R=mlp_activate(NN,X);           % Network response on this new data 
[~,Rn]=max(R,[],2);             % Response Category (1..10) &
E=(Rn-1)~=N;                    % Response accuracy per each pattern

%% ROTATION SCALE
A10=round(A/10);                % -50 .. 50 degrees -> -5 .. 5  Turn the rotation to a scale with few levels
amin=min(A10);                  % highest level in the new scale
amax=max(A10);                  % lowest level in the new scale
range10=amin:amax;              % rotation scale in the new category units
range=range10*10;               % rotation scale in the original degrees (-50 .. 50) 
na=amax-amin+1;                 % Number of new rotation levels
EA=zeros(na,1);                 % Storage for average response for each rotation-level

%% MEASURE THE ERROR
for i=1:na
  EA(i)=mean(E(A10==range10(i))); % Average response error for each rotation level
end

%% PLOT
h_fg=figure(12); clf reset;     % Set figure properies (position, size, background, .. ) ready to save as image	
set(h_fg,'Position',[500,5000,400,400],'Renderer','zbuffer','Color',[1 1 1],'PaperPositionMode', 'auto');
 plot(range,EA);                % Show the response as a function of rotation
 title('Effect of rotation');
 ylabel('Response error');
 xlabel('Rotation (degrees)');
 ylim([0 max(EA)*1.1]);
print(h_fg,'-dpng','-painters','-r100','rotation.png'); % Save the response profile as an image "response.png"
end
