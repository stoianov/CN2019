function N=dbn_train(N,X,trainepochs)

% Learning params  
N.par.mom=0.08;
N.par.eps=0.03;
N.par.dec=0.0003;
N.par.batchsize=20;
trainingpatterns=size(X,1);
N.trainingpatterns=trainingpatterns;
N.trainepochs=trainepochs;

for layer=2:N.nlayers,
  fprintf('TRAIN Layer %d ',layer);  
  for epoch=1:trainepochs
    Idata=1:trainingpatterns;		% Index of all training patterns. We will use it to create balanced batches
    while ~isempty(Idata)
 	  npat=numel(Idata);    
      if npat>N.par.batchsize
        Itr=randperm(npat,N.par.batchsize); % Index of a subset of the training data
      else
        Itr=1:npat;  				% or use all available patterns
      end
      Xpat=X(Idata(Itr),:);         % Get the training patterns
      
      noise=randn(size(Xpat))*0.02;  % Generate gaussian noise
      Xpat=max(0,min(1,Xpat+noise));% Add the noise to the input, and rectify [0-1]
      
      N=rbn_train(N,layer,Xpat);    % Train the layer on this pattern    
      Idata(Itr)=[];				% Remove the patterns we just have used
    end
    if ~rem(epoch,20), fprintf('.'); end % Print a dot every 100 epochs    
  end
  fprintf('DONE\n');
  X=layer_activate(N.W{layer},N.B{layer},X); % Activate the layer on the data, to produce inputs for the next layer
end

end
