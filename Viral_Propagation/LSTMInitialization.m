% Author: Joe Zavorskas

%%% INPUTS:

% Batch : mini-batch size for LSTM training
% NumLSTM : number of hidden layers for LSTM training
% Dropout : probability of information dropout in the dropout layer (%)
% Epochs : number of iterations of through all training data
% LearnRate : initial learning rate for backpropagation adjustments
% XTrain : Training inputs
% YTrain : Training outputs

%%% OUTPUTS:

% layers : MATLAB object representing the various layers of the LSTM
% options : MATLAB structure holding all options, including user-input
%           and defaults.

function [layers,options] = LSTMInitialization(Batch,NumLSTM,Dropout,Epochs,LearnRate,XTrain,YTrain)

numFeatures = size(XTrain{1},1);
numResponses = size(YTrain{1},1);

% Put together the lstm sandwich. Input layer takes
% in the features (sensor output), lstm layer is where
% the magic happens.
% 50 fully connected layers are used to put the data
% back together. A dropout layer is used with a dropout
% probability of 25%. (Reduces overfitting.)
% After dropout, another fully connected layer is used,
% which feeds directly into the final regression layer.
layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(NumLSTM,'OutputMode','sequence') % This means that the LSTM will output the entire time sequence.
    fullyConnectedLayer(10)
    dropoutLayer(Dropout)
    fullyConnectedLayer(numResponses)
    regressionLayer];

% Options again! I'm gonna copy/paste this from Lotka-Volterra for now,
% I'll change it later if it doesn't work.
options = trainingOptions('adam', ...
    'MaxEpochs',Epochs, ...
    'MiniBatchSize',Batch, ...
    'InitialLearnRate',LearnRate, ...
    'GradientThreshold',1, ...
    'Shuffle','never', ...
    'Verbose',true);
%     'Plots','training-progress');

end
