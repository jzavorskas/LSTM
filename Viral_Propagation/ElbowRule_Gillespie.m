function [RMSEData, RSqData] = ElbowRule_Gillespie(XTrain,YTrain,layers,options,init,k)
clc
%% Info about this File:
% Written by: Joe Zavorskas
% Started: 8/10/2021
% Last Edit: 8/11/2021

% This file will contain some copy/pasted elements from ViralPropLSTM.m.
% This file is intended to train and test the LSTM multiple times, to
% determine what is an acceptable number of data replicates. 25 Gillespie
% simulations will be generated and input. A neural network will be trained
% on subsets of the data (each of which will include more replicates). The
% RMSE and R^2 value between the LSTM prediction and numerical solution
% will be calculated for each neural network.

disp('Command Window Cleared. Begin Training.')

ReplicateCount = [1:1:25]; % 25 subsets will be created, each including one extra replicate.

ScaleVal = [20; 200; 10000];

% time points that will be used for all numerical solutions.
tspaninit = [0:10:50];
tspanrest = [75:25:200];
tspan = [tspaninit tspanrest];

% Setup for progress bar.
format = 'Training Iteration: %d \n';

RMSEData = zeros(length(ReplicateCount),3);
RSqData = zeros(length(ReplicateCount),3);

%% Master Loop here
for Replicate = 1:length(ReplicateCount)
    
    %% Training
    fprintf(format,Replicate);
    
    % Train a neural network here first.
    net(Replicate) = trainNetwork(XTrain(1:Replicate), ...
                                       YTrain(1:Replicate), layers, options);

    %% Testing
    % Running ODE. Output: three time-series (tem, gen, struct).
    % Assemble a vector with same dimensions as training input.
    [tTest,C] = ode45(@(t,C) VPDiffEq(t,C,k),tspan,[init 0 0]);
    XTestHold = zeros(length(tTest),4);
    XTestHold(:,2) = init;
    XTestHold(:,1) = tTest;
    XTest = XTestHold';
    
    % Save numerical solution to use for comparison.
    YTest = C;

    YPredTemplate = predict(net(Replicate),XTest,'MiniBatchSize',1);

    for i = 1:length(YPredTemplate(:,1))

        YPredTemplate(i,:) = YPredTemplate(i,:)*ScaleVal(i);

    end
    
    %% Analysis: RMSE
    RMSETem = sqrt(mean((YTest(:,1)' - YPredTemplate(1,:)).^2));
    RMSEGen = sqrt(mean((YTest(:,2)' - YPredTemplate(2,:)).^2));
    RMSEStruct = sqrt(mean((YTest(:,3)' - YPredTemplate(3,:)).^2));
    
    RMSEData(Replicate,:) = [RMSETem RMSEGen RMSEStruct];
    
    %% Analysis: R-Squared
    RSqTem = 1 - (sum((YTest(:,1)' - YPredTemplate(1,:)).^2))/(sum((YTest(:,1)' - mean(YTest(:,1))).^2));
    RSqGen = 1 - (sum((YTest(:,2)' - YPredTemplate(2,:)).^2))/(sum((YTest(:,2)' - mean(YTest(:,2))).^2));   
    RSqStruct = 1 - (sum((YTest(:,3)' - YPredTemplate(3,:)).^2))/(sum((YTest(:,3)' - mean(YTest(:,3))).^2));
    
    RSqData(Replicate,:) = [RSqTem RSqGen RSqStruct];
end

end
