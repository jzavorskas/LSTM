%% Info About this File:
% Written by: Joe Zavorskas
% Start: 2/14/2021
% Last Edit: 2/22/2021

% This file is used alongside brlA_Transcript_LSTM to perform manipulate
% available data and perform validation of an LSTM trained on dynamic
% transcriptomic data. The purpose of this function is to remove all data
% related to the dyanmic responses at a single micafungin concentration, to
% perform validation of the LSTM.

%%% INPUTS:

% MicaConc : The micafungin concentration whose dynamic data is to be 
%            deleted from the dataset.
% NumReps : from main file, the number of biological replicates that should
%           be included (among the micafungin concentrations that remain)
% XTrain : current cell of LSTM training inputs
% YTrain : current cell of LSTM training outputs

%%% OUTPUTS

% NewXDataCell : current cell of LSTM training inputs
% NewYDataCell : current cell of LSTM training outputs
% TrueAverage : a matrix of the transcriptional response at each micafungin
%               concentration with all biological and technical replicates
%               averaged together.

function [NewXDataCell,NewYDataCell,TrueAverage] = BioRepDataManipulation(MicaConc,NumReps,XTrain,YTrain)

%% Data Manipulation: Remove Testing Concentration and Select Bio Reps.

% Copy the original data so it can be manipulated.
XTrainCopy = XTrain;
YTrainCopy = YTrain;

% Delete the data for the testing micafungin concentration.
XTrainCopy(((15*MicaConc)+1):15*(MicaConc+1)) = [];
YTrainCopy(((15*MicaConc)+1):15*(MicaConc+1)) = [];

% 3 Tech Reps, 3 or 4 Micafungin Concs, x Bio Reps
NewXDataCell = cell(1,12*NumReps); 
NewYDataCell = cell(1,12*NumReps);

Iter = 1;
WhichReps = zeros(NumReps,1);

while WhichReps(end) == 0

    % Generate a random index (biorep number) and check if it is in the
    % list.
    RandomIndex = randi([1 5]);
    if any(WhichReps(:) == RandomIndex)
        
        continue % If it is in the list, skip the iteration and generate a new one.

    else
        
        % If it isn't, add it to the list and increment the index iterator.
        WhichReps(Iter) = RandomIndex; 
        Iter = Iter + 1;

    end

end

WhichReps = sort(WhichReps) % Sort the array for ease of use later.

%%% First, calculate the true average of the dataset.

AvgCell = cell(3*NumReps,1);
Iter = 1; % This Iter is used for the output cell.

% Use two Iterators to index the YTrain matrix.
for MidiIter = 1:NumReps

    CurrentBio = WhichReps(MidiIter);

    for MiniIter = 1:3
        
        % Data is organized as follows. There is a new micafungin 
        % concentration every 15 entries; inside which there is a new
        % biological replicate every 3 entries. Indexing below captures
        % this.
        TrainIndex = (MicaConc*15)+((CurrentBio-1)*3)+MiniIter;
        AvgCell(Iter) = YTrain(TrainIndex);
        Iter = Iter + 1; % Increment to new row in AvgCell.

    end    

end

AvgMat = cell2mat(AvgCell); % Vertically concatenate AvgCell into a matrix.

% Average each column of "AvgMat".
for i = 1:length(AvgMat(1,:))
    TrueAverage(i) = mean(AvgMat(:,i));
end

%%% To keep the same organization scheme as the original XTrain cell, input
%%% the data in the following order: Micafungin Conc, Bio Rep, Tech Rep
%%% using three nested for loops.

for BigIter = 1:4 % 3 or 4 micafungin concentrations, hard-coded

    for MidiIter = 1:NumReps % Chosen number of biological replicates

        CurrentBio = WhichReps(MidiIter);

        for MiniIter = 1:3 % Tech reps

            NewCellIndex = ((BigIter-1)*(NumReps*3))+((MidiIter-1)*3)+MiniIter;
            TrainIndex = ((BigIter-1)*15)+((CurrentBio-1)*3)+MiniIter;
            
            if [XTrainCopy{TrainIndex}(2,1)] == 10
                NewXDataCell{NewCellIndex} = [XTrainCopy{TrainIndex}(:,1:2),XTrainCopy{TrainIndex}(:,4:end)];
                NewYDataCell{NewCellIndex} = [YTrainCopy{TrainIndex}(:,1:2),YTrainCopy{TrainIndex}(:,4:end)];
            else
                NewXDataCell(NewCellIndex) = XTrainCopy(TrainIndex);
                NewYDataCell(NewCellIndex) = YTrainCopy(TrainIndex);
            end 

        end

    end

end

end