%%
% Author: Joe Zavorskas
%
%%% INPUTS:

% t : current ODE solver time
% C : vector of current concentrations of viral components;
%       1) template, 2) genome, 3) structural proteins
% k : vector of kinetic constants

%%% OUTPUTS:

% dCdt : calculated "right-hand-side" of the system of differential
%        equations representing the rate of change of each viral component

% Right-hand-side calculation function for the viral propagation model.

function dCdt = VPDiffEq(t,C,k)

dCdt = zeros(3,1);

dCdt(1) = k(1)*C(2) - k(2)*C(1);
dCdt(2) = k(3)*C(1) - k(1)*C(2) - k(4)*C(2)*C(3);
dCdt(3) = k(5)*C(1) - k(6)*C(3) - k(4)*C(2)*C(3);

end
