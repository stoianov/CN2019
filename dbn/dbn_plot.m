function dbn_plot(N,imagesize,fign)

for l=2:N.nlayers
  if l==2, 
    % The weights of the 1st computational layer represent receptieve fields
    % of each unit in the sensory domain
    W=N.W{2};      
  else
    % Recurrently multiply the weights from the lower layer by the current weights to obtain
    % an approximated sensitivity of each unit (receptieve fields) in the sensory domain
    W=W*N.W{l};
  end
  layer_plot(W,imagesize,fign+l);
end
    
end


function layer_plot(W,imagesize,fign)

figure(fign);
n=size(W,2);        % Number of units to plot
nx=round(sqrt(n));  % How many collumns of plots
ny=ceil(n/nx);      % How many rows of plots

for i=1:n,
 subplot(ny,nx,i);  % Define plot space
 w=W(:,i);          % The weights to plot as a vector
 IMG=reshape(w,imagesize); % Reshape the weights as an image
 imagesc(IMG);      % Plot the image
 axis image; axis off; 
 title(sprintf('Unit %d',i)); 
end
colormap gray;

end