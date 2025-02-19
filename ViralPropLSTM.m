function ViralPropLSTM()
%% Info about this File:
% Written by: Joe Zavorskas
% Date Started: 7/7/2021
% Last Changed: 7/28/2021

% This file is an amalgamation of all Viral Propagation work until this
% point, to make it much more readable, and functionalize the program much
% more. I will separate out the data generation and LSTM initialization
% code to declutter the information.

% Follow individual section descriptions to better understand the flow of
% this program.

%% Data Generation

Replicates = 25;
TemNoise = 0; % Fixed "noise" value, plus or minus this. 
init = 7; % Initial value of tem for input.
NumberDataSets = 1;
BlockSize = 3;
ExperimentCount = true; % This will determine whether the program jumps to new code,
                        % new code will show RMSE value for number of
                        % experiments from 1 to 10.

% Rate constants for each viral propagation "reaction".
k(1) = 0.025;
k(2) = 0.25;
k(3) = 1;
k(4) = 7.5e-6;
k(5) = 1000;
k(6) = 1.99;

% The steady state values of [tem, gen, struct], used to scale values
% before training the LSTM, such that all values are between 0 and 1.
ScaleVal = [20; 200; 10000];

% [XTrain,YTrain] = ViralPropDataGeneration(Replicates,TemNoise,k,BlockSize,NumberDataSets);
[XTrain,YTrain] = ViralPropGillespieData(Replicates,BlockSize,NumberDataSets);
    
%% Data Sparsification

% Transform Gillespie simulations from hundreds of thousands of
% time points to 12.
[XTrainSparse, YTrainSparse] = GillespieSparse(XTrain,YTrain);

% Plotting an example trajectory.
figure;

subplot(2,1,1)
plot(XTrain{1}(1,:),YTrain{1}(1,:).*20)
xlabel('Days since Infection')
ylabel('# of Template Molecules')
title('Sample Gillespie Simulation')

subplot(2,1,2)
plot(XTrainSparse{1}(1,:),YTrainSparse{1}(1,:).*20)
xlabel('Days since Infection')
ylabel('# of Template Molecules')
title('Sample Gillespie Simulation, Sparse')

%% LSTM Initialization

miniBatchSize = 1;
% How many LSTM layers?
numHiddenUnits = 10;
% Prevents overfitting, chance that data drops out during training.
dropoutChance = 0.05;
% How many times should the full dataset be used for training?
maxEpochs = 1000;
% Learning rate, slower usually gives better fit but takes more epochs.
initLearnRate = 0.01;

[layers,options] = LSTMInitialization(miniBatchSize,numHiddenUnits, ... 
                                      dropoutChance,maxEpochs,initLearnRate, ...
                                      XTrain,YTrain);
                
%% RMSE/RSq Data and Visualization

[RMSEData, RSqData] = ElbowRule_Gillespie(XTrainSparse,YTrainSparse, layers, options, init,k); 
      
%% Error Visualization

ErrorVisualization(RMSEData,RSqData)


%% LSTM Training

net = trainNetwork(XTrainSparse,YTrainSparse,layers,options);

save 'net' net

% If using moving block bookstrapping, use code below.
% for BagIter = 1:NumberDataSets
% 
%     net(BagIter) = trainNetwork(XTrain(BagIter,:),YTrain(BagIter,:),layers,options);
% 
%     save 'net' net
% 
% end

%% LSTM Testing

if NumberDataSets == 1

    tem0test = 8; % Now, interpolate a trajectory at a given value.

    tspaninit = [0:10:50]; % More data points earlier in time span.
    tspanrest = [75:25:200]; % Less later on.
    tspan = [tspaninit tspanrest];

    % Running ODE. Output: three time-series (tem, gen, struct).
    % This provides the analytical solution.
    [tTest,C] = ode45(@(t,C) VPDiffEq(t,C,k),tspan,[tem0test 0 0]);

    load 'net' net
    XTestHold = zeros(length(tTest),4); % Create a test input for LSTM.
    XTestHold(:,2) = tem0test; % Repeated initial value...
    XTestHold(:,1) = tTest; % for every time point.

    XTest = XTestHold';

    YTest = C; % Save the analytical solution here.

    % Use the LSTM to predict the time-course solution at a given
    % initial tem value.
    YPredTemplate = predict(net,XTest,'MiniBatchSize',1);

    for i = 1:length(YPredTemplate(:,1))

        YPredTemplate(i,:) = YPredTemplate(i,:)*ScaleVal(i);

    end
end
    
%% Use Each Neural Network to generate a guess.

%%% Depricated, but still useful. This section was previously used to
%%% perform bagging (bootstrap aggregation) by fitting an ensemble of 
%%% LSTMs, each to a subset of the dataset and aggregate their results
%%% for a more accurate prediction.
if NumberDataSets > 1
    
    tem0test = 3; % Now, interpolate to a value.

    tspandynam = [0:0.1:5];
    tspaninit = [6:1:20];
    tspanrest = [21:5:200];
    tspan = [tspandynam tspaninit tspanrest];
    
    load 'net' net

    YPredTemplate = zeros(length(tspan),3);
    
    [tTest,C] = ode45(@(t,C) VPDiffEq(t,C,k),tspan,[tem0test 0 0]);

    XTestHold = zeros(length(tspan),4);
    XTestHold(:,2) = tem0test;
    XTestHold(:,1) = tspan;

    XTest = XTestHold';

    YTest = C;
    
    YPredTemplateDirty = zeros(3,length(YPredTemplate(:,1)),NumberDataSets);

    for Test = 1:NumberDataSets

        YPredTemplateDirty(:,:,Test) = predict(net(Test),XTest,'MiniBatchSize',1);

    end

    
    YPredTemplate = mean(YPredTemplateDirty,3);
    
    for Unscale = 1:3

        YPredTemplate(Unscale,:) = YPredTemplate(Unscale,:)*ScaleVal(Unscale);
        
    end
    
end

%% Visualization

figure;
subplot(2,2,1)
hold on
plot(tTest,YPredTemplate(1,:)','.-')
plot(tTest,YTest(:,1),'--')
hold off
xlabel('Time (days post-infection)')
ylabel('tem molecules')
legend('Prediction','Numerical Solution')
title('Comparison of Numerical Solution and LSTM Prediction, Viral Propagation')

subplot(2,2,2)
hold on
plot(tTest,YPredTemplate(2,:)','.-')
plot(tTest,YTest(:,2),'--')
hold off
xlabel('Time (days post-infection)')
ylabel('tem molecules')
legend('Prediction','Numerical Solution')
title('Comparison of Numerical Solution and LSTM Prediction, Viral Propagation')

subplot(2,2,3)
hold on
plot(tTest,YPredTemplate(3,:)','.-')
plot(tTest,YTest(:,3),'--')
hold off
xlabel('Time (days post-infection)')
ylabel('tem molecules')
legend('Prediction','Numerical Solution')
title('Comparison of Numerical Solution and LSTM Prediction, Viral Propagation')


%% Data Output for GPTIPS

%%% Also depricated, this project was originally going to support
%%% a genetic program, but that part of the project was discontinued.
DataOut = zeros(length(tspan),4);
DataOut(:,1) = tspan;
DataOut(:,2:end) = YPredTemplate';

writematrix(DataOut, 'VPData.csv')

end