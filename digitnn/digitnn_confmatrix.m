function [CM,err]=digitnn_confmatrix(R,N)

 n=size(R,1);                   % Number of test cases
 C=N+1;                         % True category of each pattern (1..10)
 [~,Rn]=max(R,[],2);            % Perceived category (1..10)
 err=mean((Rn-1)~=N);           % Overall error

 CM=zeros(10,10);               % Confusion matrix: distribution of perceived categories (columns) for each true category (rows)
 
 for i=1:n                      % get each pattern
    CM(C(i),Rn(i)) = CM(C(i),Rn(i)) + 1; 
    % For each pattern, count the perceived category of each pattern; 
    % Rows are true categories and columns are perceived categories
 end
 CC=sum(CM,2);                  % Total counts per each category
 CM=CM./repmat(CC,1,10);        % Empirical distribution of category assignment
end