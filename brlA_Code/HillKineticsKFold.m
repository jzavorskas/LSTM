function HillKineticsKFold()

Times = [0 10 20 30 60 90];
Micas = [0 5 10 15 20];
NumBioReps = 5;
Ydata2D = readmatrix("brlA_Data_Master.xlsx",'Sheet','AvgTechReps');

K = 5;
% % Set random number generation to a default value for reproducibility.
rng('default')
% rng('shuffle')
% Define a partition element that will generate 5 folds on all data points.
KFoldData = cvpartition(150,'KFold',K);

YdataFlat = zeros(150,1); % Response
XdataFlat = zeros(150,2); % Dose, Time

IDCount = 0;

for i = 1:length(Micas) % length = 5

    for j = 1:NumBioReps % length = 5

        for k = 1:length(Times) % length = 6

            IDCount = IDCount + 1;

            XdataFlat(IDCount,1) = Micas(i);
            XdataFlat(IDCount,2) = Times(k);
            YdataFlat(IDCount,1) = Ydata2D(((i-1)*5)+j,k);

        end

    end

end

% model = @(p, x) ...
%     1 + ((p(2) * x(:,1)) ./ (p(1) + x(:,1) + (x(:,1).^2 / p(3))) .* ...
%     (1 - exp(-p(4) * x(:,2))));

model = @(p, x) ...
     1 + ((p(2) * x(:,1)) ./ (p(1) + x(:,1)) .* ...
     (1 - exp(-p(4) * x(:,2))));

% Km, Vmax (maximum response), Ki, time constant (m^-1)
p0 = [10, 5, 1, 0.5]; 

options = optimoptions('lsqcurvefit', ...
                       'MaxFunctionEvaluations', 10000, ...
                       'MaxIterations', 10000, ...
                       'StepTolerance', 1e-12, ...
                       'OptimalityTolerance', 1e-12, ...
                       'FunctionTolerance', 1e-12);

TrueAverage = [];
RMSETrueAverage = [];
RSqtoAvg = [];

for Fold = 1:K

    % Output the current run (fold) number.
    format1 = 'Currently training fold iteration %d.';
    sprintf(format1,Fold)

    % Generate the ids of the training set.
    idxTrainDelete = training(KFoldData,Fold);
    idxTestDelete = test(KFoldData,Fold);

    XTrain = XdataFlat; YTrain = YdataFlat;

    % Copy XTrain to a second variable so that we can delete from both in
    % parallel. Create the testing set for the inputs and outputs.
    XTest = XTrain; YTest = YTrain;

    % Delete training values from the testing set.
    XTest(idxTrainDelete,:) = [];
    YTest(idxTrainDelete,:) = [];
    
    % Now, delete any testing points from the training set.
    XTrain(idxTestDelete,:) = [];
    YTrain(idxTestDelete,:) = [];

    [best_p, resnorm,residual,exitflag,output] = lsqcurvefit(model, p0, XTrain, YTrain, ...
                                    [0,0,0,0], [25, max(YTrain), inf, inf], options);
    
    % Testing:
    YPred = zeros(length(YTest),1);
    TrueAverage = zeros(length(YTest),1);
    TestData = zeros(length(YTest),2);

    for TestRep = 1:length(YTest)

        % Grab the index of the time point that the test point is sampling
        % at, then pull that column from the table of averaged bio/tech
        % reps.
        FindIndexDose = find(Micas == XTest(TestRep,1));
        FindIndexTime = find(Times == XTest(TestRep,2));
        TrueAverage(TestRep) = AveragedMicas(FindIndexDose,FindIndexTime);
        TestData(TestRep,:) = [FindIndexDose,FindIndexTime];

%         YPred(TestRep) = 1 + (((best_p(2) * XTest(TestRep,1)) ./ (best_p(1) + XTest(TestRep,1) + ...
%                 (XTest(TestRep,1).^2 / best_p(3)))) .* ...
%                 (1 - exp(-best_p(4) * XTest(TestRep,2))));

        YPred(TestRep) = 1 + (((best_p(2) * XTest(TestRep,1)) ./ (best_p(1) + XTest(TestRep,1))) .* ...
                 (1 - exp(-best_p(4) * XTest(TestRep,2))));

    end

    RMSETrueAverage(Fold) = sqrt(mean((TrueAverage - YPred).^2));

    test1 = (TrueAverage-YPred).^2;
    test2 = (TrueAverage-mean(TrueAverage)).^2;
    RSqtoAvg(Fold) = 1-(sum(test1)/sum(test2));

end

end