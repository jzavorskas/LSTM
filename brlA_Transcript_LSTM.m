function brlA_Transcript_LSTM()
%% Info About this File:
% Written by: Joe Zavorskas
% Start: 2/14/2021
% Last Edit: 2/22/2021

% This file will adapt the LSTM code written in July-August, 2021 to the
% brlA transcriptomics data collected by Zavorskas, J. Edwards, H. in
% January 2022. 75 time courses of brlA regulation are available to train
% on, at 5 different micafungin concentrations (including zero). This
% program will train on 2 or 3 micafungin concentrations in full to attempt
% to predict/generate the remaining concentrations.

% Note: the performance of this program will be determined by comparing
% the prediction of the LSTM for a certain micafungin concentration to the
% average of all replicates for that micafungin concentration. Synthetic
% data is valuable if it follows the trend of previously produced data and
% loosely captures fungal behavior.

% I am going to code long-form for most of this to spread out the different
% test cases (all interpolation, one interpolate/one extrapolate, all
% extrapolation).

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

%% LSTM Initialization

miniBatchSize = 5;
% How many LSTM layers?
numHiddenUnits = 10;
% Prevents overfitting, chance that data drops out during training.
dropoutChance = 0.2;
% How many times should the full dataset be used for training?
maxEpochs = 875;
% Learning rate, slower usually gives better fit but takes more epochs.
initLearnRate = 0.000075;

[layers,options] = LSTMInitialization(miniBatchSize,numHiddenUnits, ... 
                                      dropoutChance,maxEpochs,initLearnRate, ...
                                      XTrain,YTrain);

%% Data Manipulation: Remove Testing Concentration and Select Bio Reps.

Times = [0 10 20 30 60 90];
Micas = [0 5 10 15 20];

MicaConc = 2; %%% 0: 0ng/mL, 1: 5ng/mL, 2: 10ng/mL, 3: 15ng/mL, 4: 20ng/mL.
% TimePoint = 4; %%% 0: 0m, 1: 10m, 2: 20m, 3: 30m, 4: 60m, 5: 90m.
NumReps = 5; %%% How many bio reps should the neural net train on?
Runs = 2; %%% How many times should the network train and calculate R^2?

RsqRuns = zeros(Runs,1);

for Run = 1:Runs

    format = 'Run %f Started.';
    sprintf(format,Run)

    % Use external function to create a new X- and Y-Train, by removing the 
    % micafungin concentration of interest, pulling out "NumReps" random
    % bio replicates, and finding the true average of the testing set.
     [NewXDataCell,NewYDataCell,TrueAverage] = BioRepDataManipulation(MicaConc,NumReps,XTrain,YTrain);
%    [NewXDataCell,NewYDataCell,TrueAverage] = BioRepDataManipulationDR(TimePoint,NumReps,XTrainDR,YTrainDR);

    net = trainNetwork(NewXDataCell,NewYDataCell,layers,options);
    
    XTest = [0 10 20 30 60 90; MicaConc*5 MicaConc*5 MicaConc*5 MicaConc*5 MicaConc*5 MicaConc*5];
%    XTestDR = [0 5 10 15 20; Times(TimePoint+1) Times(TimePoint+1) Times(TimePoint+1) Times(TimePoint+1) Times(TimePoint+1)];
    YPred = predict(net,XTest,'MiniBatchSize',1);
%    YPredDR = predict(net,XTestDR,'MiniBatchSize',1);

    test1 = (TrueAverage-YPred).^2;
    test2 = (TrueAverage-mean(TrueAverage)).^2;
    RsqRuns(Run) = 1-(sum(test1)/sum(test2))
    
end

% Print the current accuracy metrics.
MeanRsq = mean(RsqRuns)
StDevRsq = std(RsqRuns)

figure;
hold on
plot(XTest(1,:),TrueAverage)
plot(XTest(1,:),YPred)
hold off
legend('True','Predict')

%_________________________________________________________________________

%% K-Fold Cross Validation, Bio Dataset

K = 75;
% % Set random number generation to a default value for reproducibility.
rng('default')
% rng('shuffle')
% Define a partition element that will generate 5 folds on 75 data points.
KFoldData = cvpartition(75,'KFold',K);

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
    net = trainNetwork(XTrain,YTrain,layers,options);
    
    % Iterate through all testing points, calculating the RMSE.
    for TestRep = 1:length(YTest)

        % Grab the index of the time point that the test point is sampling
        % at, then pull that column from the table of averaged bio/tech
        % reps.
        FindIndex = find(Micas == XTest{TestRep}(2,1));
        TrueAverage = AveragedMicas(FindIndex,:);

        % Predict each point in the XTest field, and calculate the RMSE
        % between the prediction and the known value.
        YPred = predict(net,XTest{TestRep},'MiniBatchSize',1);
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

    format2 = 'Run: %d. Individual RMSE: %2.3f. RMSE to Average: %2.3f. RSquared to Average: %2.3f';
    sprintf(format2,Fold,AverageRMSE(Fold),AverageRMSEtoAvg(Fold),AverageRsq(Fold))

end

%% Surface Plot Generation

% Preallocate arrays that will hold the range of values to sample over.
% Sample every 1 ng/mL on micafungin concentration, every 2.5 minutes on
% time.
MicaSample = linspace(0,20,21);
TimeSample = linspace(0,90,37);

% Train LSTM on one of the datasets. I'm going to tune the settings a
% little while I do this to hopefully make the LSTM fit well.
SurfaceNet = trainNetwork(XTrain,YTrain,layers,options);

% 1. Train on both, test on DR curves.
% 2. Train on both, test on time courses.
XTestMC = zeros(1,6);
ZPredTime = zeros(5,length(TimeSample));
ZPredCombo = zeros(length(MicaSample),6);

% Prediction loop for dose-response curve.
for Iter = 1:length(MicaSample)
    
    XTestMC(:) = MicaSample(Iter);
    XTest = [0 10 20 30 60 90; XTestMC];
    % Query the network for each Sample.
    ZPredCombo(Iter,:) = predict(SurfaceNet,XTest,'MiniBatchSize',1);

end

save 

figure;
surf([0 10 20 30 60 90],MicaSample,ZPredCombo)
xlabel('Time (min)')
ylabel('Micafungin Concentration (ng/mL)')
zlabel('Predicted brlA Fold Change')
end