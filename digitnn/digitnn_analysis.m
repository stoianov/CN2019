function ER=digitnn_analysis(ER,R,N,A)
 n=size(R,1);                   % Number of test cases
 %% Overall error
 [~,Rn]=max(R,[],2);            % Response Category
 E=mean((Rn-1)~=N);             % Overall error
 ER=[ER E];                     % Increment the error vector

 %% Confusion matrix
 CM=zeros(10,10);               % Confusion matrix counts the assignments per class
 for i=1:n
    CM(N(i)+1,Rn(i)) = CM(N(i)+1,Rn(i)) + 1;
 end
 CC=sum(CM,2);                  % Counts per class
 CM=CM./repmat(CC,1,10);        % Empirical distribution of assignments per class
 
  %% SHOW ANALYSIS
  disp('Confusion matrix (Assignments per classes). Each row is a class');
  disp(round(CM*100));
  
  tit=sprintf('Overall Error  %.2f',ER(end));
  disp(tit);
  
  %% FIGURE with ERR and Confusion Matrix
  figure(1);  
  subplot(1,2,1); 
    plot(ER);                   % Show the learning trend
    ylim([0 1]);
    title('Learning trend');
    ylabel('Error on test set');
    xlabel('Training session');
  subplot(1,2,2); 
    imagesc(CM);                % Show the confusion matrix
    title(tit);
    ylabel('Class (0,1 .. 9)');
    xlabel('Response (0,1 .. 9)');
  drawnow
end