function [XTrain,YTrain] = ViralPropGillespieData(Replicates,BlockSize,NumberDataSets)

% Author: Joe Zavorskas

%%% INPUTS:

% Replicates : number of stochastic replicates to generate
% BlockSize : if using moving block bootstrapping, number of replicates
%             per block
% NumberDataSets : if using moving block bootstrapping, number of datasets
%                  to generate using random selection with replacement

%%% OUTPUTS:

% XTrain : generated inputs for LSTM training
% YTrain : generated outputs for LSTM training

import Gillespie.*

ScaleVal = [20; 200; 10000];

%% Reaction network:
%   1: "gen" -k1-> "tem"        (RNA Template Formation)
%   2: "tem" -k2-> 0            (Degradation)
%   3: 0 -k3-> "gen"        (Reverse Transcription)
%   4: "gen" + "struct" -k4-> 0 (Virus Formation)
%   5: 0 -k5-> "struct"     (Protein Creation)
%   6: "struct" -k6-> 0         (Secretion)


%% Rate constants
p.k1 = 0.025; % day^-1      
p.k2 = 0.25; % day^-1             
p.k3 = 1.0; % day^-1                        
p.k4 = 7.5e-6; % day^-1*molecule^-1
p.k5 = 1000; % day^-1
p.k6 = 1.99; % day^-1

%% Initial state
tspan = [0, 200]; %days
 %tem, gen, struct

%% Specify reaction network
pfun = @propensities_2state;
% Stoichiometric Matrix for viral propagation equations.
stoich_matrix = [ 1  -1  0    
                  -1  0  0    
                  0  1  0        
                  0  -1  -1
                  0  0  1
                  0   0  -1];     
              
for Iter = 1:Replicates
    
    % Print status updates for each replicate.
    format = 'Gillespie Replicate: %d \n';
    Replicate = Iter;
    fprintf(format,Replicate);
    
    XTrainHold = [];
    YTrainNoise = [];
    
    x0tem = 6 + (6-10).*rand(1,1); % random initial condition for tem
    x0 = [x0tem, 0, 0];  %tem, gen, struct
    
    [t,x] = directMethod(stoich_matrix, pfun, tspan, x0, p);
%     [t,x] = firstReactionMethod(stoich_matrix, pfun, tspan, x0, p);
    
    XTrainHold(:,1) = t; % First row of training X values is time 
    YTrainNoise(:,:) = x; % Solution of gillespie is training Y values
    
    for i = 1:length(t)
       
        % the rest of training X values is initial condition [tem, gen,
        % struct] repeated.
        XTrainHold(i,2:4) = x0; 
        
    end
    
    %%% Depricated for final project. If bootstrap aggregation is used,
    %%% this section selects subsets of data with replacement and
    %%% trains multiple LSTMs.
    if NumberDataSets > 1

        [XTrainMBB,YTrainMBB] = MovingBlockBootstrap(XTrainHold,YTrainNoise,BlockSize,NumberDataSets);

        for Dataset = 1:NumberDataSets

            XTrain{Dataset,Iter} = (XTrainMBB(:,:,Dataset))';
            YTrain{Dataset,Iter} = (YTrainMBB(:,:,Dataset))';

        end

    else 
        
        % Scale all values by steady-state maxima (20, 200, 10000)
        % so that values training LSTM are between 0 and 1.
        YTrainNoise(:,1) = YTrainNoise(:,1)/ScaleVal(1);
        YTrainNoise(:,2) = YTrainNoise(:,2)/ScaleVal(2);
        YTrainNoise(:,3) = YTrainNoise(:,3)/ScaleVal(3);
            
        XTrain{Iter} = XTrainHold';
        YTrain{Iter} = YTrainNoise';

    end
 
end
    
end


function a = propensities_2state(x, p)
% Return reaction propensities given current state x
tem = x(1);
gen = x(2);
struct = x(3);

% Calculates the rate probability of each reaction.
% only one reaction can "fire" per time step, so typically the fastest
% reaction as calculated below will "fire".
a = [p.k1*gen;            
     p.k2*tem;      
     p.k3*tem;       
     p.k4*gen*struct;
     p.k5*tem;
     p.k6*struct];   
 
end



