function ErrorVisualization(RMSEData,RSqData)

ReplicateCount = [1:1:25];

%% RMSE Visualization Plot
figure;
subplot(2,2,1)
hold on
plot(ReplicateCount,RMSEData(:,1),'.-')
hold off
xlabel('Number of Replicates')
ylabel('RMS Errors')
title('Root-Mean-Squared Errors for LSTM Predictions, Tem')

subplot(2,2,2)
hold on
plot(ReplicateCount,RMSEData(:,2),'.-')
hold off
xlabel('Number of Replicates')
ylabel('RMS Errors')
title('Root-Mean-Squared Errors for LSTM Predictions, Gen')

subplot(2,2,3)
hold on
plot(ReplicateCount,RMSEData(:,3),'.-')
hold off
xlabel('Number of Replicates')
ylabel('RMS Errors')
title('Root-Mean-Squared Errors for LSTM Predictions, Struct')

%% R-Squared Visualization Plot
figure;
subplot(2,2,1)
hold on
plot(ReplicateCount,RSqData(:,1),'.-')
hold off
xlabel('Number of Replicates')
ylabel('R-Squared')
title('R-Squared Value for LSTM Prediction, Tem')

subplot(2,2,2)
hold on
plot(ReplicateCount,RSqData(:,2),'.-')
hold off
xlabel('Number of Replicates')
ylabel('R-Squared')
title('R-Squared Value for LSTM Prediction, Gen')


subplot(2,2,3)
hold on
plot(ReplicateCount,RSqData(:,3),'.-')
hold off
xlabel('Number of Replicates')
ylabel('R-Squared')
title('R-Squared Value for LSTM Prediction, Struct')


end
