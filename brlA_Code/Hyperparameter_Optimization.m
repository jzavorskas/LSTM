function Hyperparameter_Optimization()
%% Info About this File:
% Written by: Joe Zavorskas
% Start: 12/29/2025
% Last Edit: 12/30/2025

% This file will perform hyperparameter optimzation to support LSTM code 
% that trains on 75 time courses of brlA regulation at 5 different 
% micafungin concentrations (including zero). This program will train on
% all data in each test case, and the hyperparameters 

%% Hyperparameter Optimization Grid
LSTMLayers = [1; 3; 6; 10];
FullLayers = [1; 3; 6; 10];
NumEpoch = [500, 1000, 1500];

%% Data Input and Organization
NumPoints = 75;

% Input only my full, organized sheet of time-courses.
FullTable = readmatrix("brlA_Data_Master.xlsx",'Sheet','LSTM Data Organization');

% Import the "correct", averaged micafungin trajectories for comparison to
% the LSTM's predictions.
AveragedMicasIn = readmatrix("brlA_Data_Master.xlsx",'Sheet','AllMicaAvs');
AveragedMicas = AveragedMicasIn(2:6,2:end);

% Create cells to hold the 75 time courses in the training data.
XTrain = cell(1,NumPoints);
YTrain = cell(1,NumPoints);

% Create a matrix that will hold the times and micafungin concentrations.
XTrainHold = zeros(2,6);

% Hard-coding this for now. Need to add as separate micafungin
% concentrations.
for i = 1:NumPoints
    
    % Calculate one of the micafungin concentrations: 0, 5, 10, 20, 30.
    % There are 15 data points for each, so this equation is able to
    % calculate based on iteration number. Need an exception for perfectly
    % divisible numbers.

    if mod(i,15) ~= 0
        MicaConc = floor(i/15)*5;
    else
        MicaConc = (i/15 - 1)*5;
    end

    XTrainHold(1,:) = FullTable(1,1:6);
    XTrainHold(2,:) = MicaConc;

    % Assemble XTrain and YTrain. The LSTM in MATLAB requires the
    % input/output training data to have distinct features in rows, with
    % data going across.
    XTrain{i} = XTrainHold;
    YTrain{i} = FullTable(i,7:12);
    
end

XTrainSave = XTrain;
YTrainSave = YTrain;

%% Hyperparameter Optimization Loop

HyperParamR2 = zeros(numel(LSTMLayers),numel(FullLayers),numel(NumEpoch));
HyperParamRMSE = zeros(numel(LSTMLayers),numel(FullLayers),numel(NumEpoch));

Times = [0 10 20 30 60 90];
Micas = [0 5 10 15 20];
iter = 0;

for i = 1:numel(LSTMLayers)

    for j = 1:numel(FullLayers)

        for k = 1:numel(NumEpoch)

            iter = iter + 1;
            disp(i)
            disp(j)
            disp(k)

            %% LSTM Initialization
            
            miniBatchSize = 5;
            % Learning rate, slower usually gives better fit but takes more epochs.
            initLearnRate = 0.001;
            % Prevents overfitting, chance that data drops out during training.
            dropoutChance = 0.2;
            
            % These are all determined by the optimization loop.
            % How many fully connected layers before dropout?
            numFullyConnected = FullLayers(j);
            % How many LSTM layers?
            numHiddenUnits = LSTMLayers(i);
            % How many times should the full dataset be used for training?
            maxEpochs = NumEpoch(k);
            
            disp(["Training LSTM Iteration ", iter])
            [layers,options] = LSTMInitialization_opt(miniBatchSize,numHiddenUnits, ...
                                                  numFullyConnected,dropoutChance, ...
                                                  maxEpochs,initLearnRate, ...
                                                  XTrain,YTrain);
            
            
            HyperOptnet = trainNetwork(XTrain,YTrain,layers,options);

            disp(["LSTM Training Complete Iteration ", iter])
            NumSum = 0;
            DenomSum = 0;
            MSETrueAverage = zeros(numel(Micas),1);

            for q = 1:numel(Micas)

                MicaConc = Micas(q);
                XTest = [0 10 20 30 60 90; MicaConc MicaConc MicaConc MicaConc MicaConc MicaConc];
                
                % Grab the index of the time point that the test point is sampling
                % at, then pull that column from the table of averaged bio/tech
                % reps.
                FindIndex = find(Micas == XTest(2,1));
                TrueAverage = AveragedMicas(FindIndex,:);
                
                YPred = predict(HyperOptnet,XTest,'MiniBatchSize',1);

                test1 = (TrueAverage-YPred).^2;
                test2 = (TrueAverage-mean(TrueAverage)).^2;
                
                NumSum = NumSum + test1;
                DenomSum = DenomSum + test2;

                MSETrueAverage(q) = mean((TrueAverage - YPred).^2);

            end 

            HyperParamR2(i,j,k) = 1-(sum(test1)/sum(test2))
            HyperParamRMSE(i,j,k) = sqrt(sum(MSETrueAverage))

        end
    end
end


end