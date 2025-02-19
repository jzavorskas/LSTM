function [XDataMBB,YDataMBB] = MovingBlockBootstrap(XData,YData,BlockSize,NumberDataSets)
%% Info about this File:
% Written by: Joe Zavorskas
% Date Started: 6/28/2021
% Last Changed: 6/28/2021

% This file will is an application of moving block
% bootstrapping (MBB) to the Viral Propagation model.
% This file is designed to take in time series data 
% as a two-or-more column matrix.
% (time, dependent variable 1, dependent variable 2, ...).

% The Viral Propagation LSTM is now excellent at tracking the exponential
% growth and steady state regimes, with good interpolation between 1 and 5
% "tem" concentration. However, the LSTM struggles with the initial
% depletion phase. Hopefully, MBB and aggregation of an ensemble of neural
% nets can make the initial depletion phase more accurate.

%% Preparation Section

DataCount = length(XData(:,1)); % Save original size of the data.

NumberofBlocks = DataCount/BlockSize; % How many blocks are needed?
NumberofDependent = (length(XData(1,:))) - 1; % How many dependent variables are there?

NumberofFeatures = length(YData(1,:));
NumberPossibleBlocks = DataCount - 2*floor(BlockSize/2); % How many blocks are possible in the dataset?
Offset = floor(BlockSize/2); % Number of entries from edge to center of block.

XDataHold = zeros(DataCount,NumberofDependent+1);
YDataHold = zeros(DataCount,NumberofFeatures);

XDataMBB = zeros(DataCount,NumberofDependent+1,NumberDataSets);
YDataMBB = zeros(DataCount,NumberofFeatures,NumberDataSets);

%% Bootstrapping Loop

for Iter = 1:NumberDataSets

    for i = 1:NumberofBlocks
        
       LowBound = BlockSize*i - (BlockSize-1);
       UpperBound = BlockSize*i; 
       
       % Randomly select the center point of an available block. Size must
       % be fixed, so some numbers on the ends are off limits.
       BlockCenter = randi([(1+Offset),(DataCount-Offset)]);
       
       % Find the edges of the block and place the block in the correct
       % location.
       XDataHold(LowBound:UpperBound,:) = XData((BlockCenter-Offset):(BlockCenter+Offset),:);
       YDataHold(LowBound:UpperBound,:) = YData((BlockCenter-Offset):(BlockCenter+Offset),:);
       
    end
    
    % Sort the matrix to make sure that time increases down the column.
    [XDataSort,Indexes] = sortrows(XDataHold);
    
    for i = 1:length(Indexes)
       
        YDataSort(i,:) = YDataHold(Indexes(i),:);
        
    end
    
    XDataMBB(:,:,Iter) = XDataSort;
    YDataMBB(:,:,Iter) = YDataSort;
    
end

end