function digitnn_errorplot(CM,ERR,Leg,layer,fign)
 figure(fign);  
 
 subplot(1,2,1); 
    plot(ERR');                  % Show the learning trend
    ylim([0 1]);
    title('Learning trend');
    ylabel('Error on test set');
    xlabel('Training session');
    legend(Leg);
    
 subplot(1,2,2); 
    imagesc(CM);                % Show the confusion matrix
    title(sprintf('Confusion matrix layer %d',layer));
    ylabel('True category');
    xlabel('Perceived category');
 drawnow;
end
