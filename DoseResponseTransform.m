%% Slice the data into Dose-Response Curves instead Time-Courses.

% Written by: Joe Zavorskas
% Start: 2/14/2022
% Last Edit: 2/22/2022

% This file is an extension of the LSTM code written in July-August, 2021 
% to plot static dose-response curves (one time, all dosages) instead of
% dynamic transcriptomic responses (one dosage, all times)
% data originally collected by Zavorskas, J. Edwards, H. in
% January 2022. 75 time courses of brlA dynamics are transposed into 90
% dose-response curves for use in training an LSTM.

function [XTrainDR,YTrainDR] = DoseResponseTransform(FullTable)

    % New Training cells for dose-response data.
    XTrainDR = cell(1,90);
    YTrainDR = cell(1,90);
    
    % Lay out all concentrations and times.
    MicaConcs = [0 5 10 15 20];
    Times = [0 10 20 30 60 90];
    
    % Initialize an array that will be used to assemble all Y-values
    % for the training dataset.
    YTrainInput = zeros(1,5);
    
    for Column = 1:6
    
        % for training input, assemble a 2x5 matrix with constant first row
        % containing all possible micafungin concentrations, and second row
        % containing five repeats of the current time point.
        XTrainInput = [MicaConcs; Times(Column) Times(Column) Times(Column) Times(Column) Times(Column)];
    
        for Replicate = 1:15
            % Save the training inputs in a cell.
            XTrainDR{((Column-1)*15)+Replicate} = XTrainInput;
    
            for Point = 1:5
                % Read the data table at each trajectory, transposing
                % dynamics to dose-response behavior.
                YTrainInput(Point) = FullTable(((Point-1)*15)+Replicate,Column+6);
    
            end
            % Save the transposed data as training output.
            YTrainDR{((Column-1)*15)+Replicate} = YTrainInput;
    
        end
    
    end
end