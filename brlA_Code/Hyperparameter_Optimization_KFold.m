function Hyperparameter_Optimization_KFold()
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
            
            AverageRMSE = [];
            AverageRMSEtoAvg= [];
            AverageRsq = [];

            for Fold = 1:K
            
                % Reset the looping values for accuracy.
                RMSEIndiv = [];
                RMSETrueAverage = [];
                RSqtoAvg = [];
            
                % Output the current run (fold) number.
                format1 = 'Currently training fold iteration %d.';
                sprintf(format1,Fold)
            
                XTrain = XTrainSave;
                YTrain = YTrainSave;
            
                % Generate the ids of the training set.
                idxTrainDelete = training(KFoldData,Fold);
                idxTestDelete = test(KFoldData,Fold);
                
                % Copy XTrain to a second variable so that we can delete from both in
                % parallel. Create the testing set for the inputs and outputs.
                XTest = XTrain; YTest = YTrain;
            
                % Delete training values from the testing set.
                XTest(:,idxTrainDelete) = [];
                YTest(:,idxTrainDelete) = [];
                
                % Now, delete any testing points from the training set.
                XTrain(:,idxTestDelete) = [];
                YTrain(:,idxTestDelete) = [];
                
                % This is the rate-limiting step. The network will need to be trained
                % 5 times total (or 75 in leave-one-out).
                HyperOptnet = trainNetwork(XTrain,YTrain,layers,options);
                
                % Iterate through all testing points, calculating the RMSE.
                for TestRep = 1:length(YTest)
            
                    % Grab the index of the time point that the test point is sampling
                    % at, then pull that column from the table of averaged bio/tech
                    % reps.
                    FindIndex = find(Micas == XTest{TestRep}(2,1));
                    TrueAverage = AveragedMicas(FindIndex,:);
            
                    % Predict each point in the XTest field, and calculate the RMSE
                    % between the prediction and the known value.
                    YPred = predict(HyperOptnet,XTest{TestRep},'MiniBatchSize',1);
                    RMSEIndiv(TestRep) = sqrt(mean((YTest{TestRep} - YPred).^2));
                    RMSETrueAverage(TestRep) = sqrt(mean((TrueAverage - YPred).^2));
            
                    test1 = (TrueAverage-YPred).^2;
                    test2 = (TrueAverage-mean(TrueAverage)).^2;
                    RSqHold = 1-(sum(test1)/sum(test2));
                    RSqtoAvg = [RSqtoAvg RSqHold];
            
                end
            
                % Calculate and output the averaged RMSE values for each run.
                AverageRMSE(Fold) = mean(RMSEIndiv);
                AverageRMSEtoAvg(Fold) = mean(RMSETrueAverage);
                AverageRsq(Fold) = mean(RSqtoAvg);
            
                format2 = 'Fold: %d. Individual RMSE: %2.3f. RMSE to Average: %2.3f. RSquared to Average: %2.3f';
                sprintf(format2,Fold,AverageRMSE(Fold),AverageRMSEtoAvg(Fold),AverageRsq(Fold))
            
            end
           
            HyperParamR2(i,j,k) = mean(AverageRsq)
            HyperParamRMSE(i,j,k) = mean(AverageRMSEtoAvg)

        end
    end
end

end