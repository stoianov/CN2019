% Plot the weights of the hidden units as images
function plot_weights(NN)

h_fg=figure(2);clf reset;
set(h_fg,'Position',[100,1000,1500,800],'Renderer','zbuffer','Color',[1 1 1],'PaperPositionMode', 'auto');

clim=2;                         % Limits of color scale, to show more clearly the response patterns
cthr=1;                         % Plot as Zero any (absolute) weight-value bellow this threshold
npl=NN.sz(2); nx=10; ny=npl/10; % How many images weight matrixes to show

for i=1:npl,                    % Each hidden unit
  subplot(ny,nx,i);             % Subplot for this unit
  w=NN.hw(:,i);                 % the weight matrix of this unit as a vector
  w(abs(w)<cthr)=0;             % Clear any absoloute weight value bellow the threshold
  img=reshape(w,20,20);         % Reshape the weight matrix as an image according to the original image size
  imagesc(img);                 % Show the image with the weight matrix
  axis image; axis off;         % Set the type of plot as "image" and remove axes for clear visualization
  caxis([-clim clim]);          % Set the color limits
  s=sprintf('unit %d',i);       % The classes are 0..9 
  text(1,-3,s);
end
colormap bone;                  % Use the so-called "bone" color scale which is a type of grasyscale 

print(h_fg,'-dpng','-painters','-r100','HL.png'); % Save the figure as an image "HL.png"
end

