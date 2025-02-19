function [XSparse,YSparse] = GillespieSparse(X,Y)
%% Info about this File:
% Written by: Joe Zavorskas
% Started: 8/10/2021
% Last Edit: 8/10/2021

% This is a short program that will transform the incredibly dense
% gillespie simulation data into data that looks more like previous A.
% nidulans experiments. This means cutting down from millions of data
% points to between 10-15.

DataPoints = [0 10 20 30 40 50 75 100 125 150 175 200]; % 12 data points.

% Preallocate 12 data spaces in the output matrixes.
XSparse = {};
YSparse = {};

% These will check on the "target" time points, and see when they've been
% reached.
CheckPointIndex = 1;
CheckPoint = DataPoints(CheckPointIndex);

% Loops through every available time point, stopping when the current time
% passes a time checkpoint.
for Set = 1:length(X(1,:))
    
    % Split up an input cell matrix into the time points and initial
    % conditions. We only need the time points for the loop.
    TimePoints = X{Set}(1,:);
    XData = X{Set}(2:4,:);
    
    CheckPointIndex = 1;
    CheckPoint = DataPoints(CheckPointIndex);
    
    for t = 1:length(TimePoints)

        if TimePoints(t) > CheckPoint

            % Since the checkpoint has already been passed, take the previous
            % time point at this step.
            XSparse{Set}(:,CheckPointIndex) = X{Set}(:,t-1);
            YSparse{Set}(:,CheckPointIndex) = Y{Set}(:,t-1);

            % Update to a new time point and new index.
            CheckPointIndex = CheckPointIndex + 1;
            CheckPoint = DataPoints(CheckPointIndex);

        end

    end
    
    % The t = 200 days point will always be missed by this code. Need to
    % manually append it to the dataset.
    XSparse{Set}(:,CheckPointIndex) = X{Set}(:,end);
    YSparse{Set}(:,CheckPointIndex) = Y{Set}(:,end);
    
end

end
