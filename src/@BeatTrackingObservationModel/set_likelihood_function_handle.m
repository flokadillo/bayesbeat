function [] = set_likelihood_function_handle(obj)
% set_likelihood_function_handle(obj)
% Sets the function handle to compute the observation likelihood
%
% ------------------------------------------------------------------------
%INPUT parameters:
% obj
%
% 27.08.2015 by Florian Krebs
% ------------------------------------------------------------------------
if ~ismember(obj.dist_type, {'MOG', 'RNN'})
   error('Unknown distribution %s for observation likelihood\n', obj.dist_type); 
end
obj.compute_likelihood = str2func(obj.dist_type);
end

function O = MOG(observation, obs_params)
% Mixture of Gaussians (MOG)
% obs_params.obs:  parameters of gamma distribution for each state [states x 2]
O = zeros(length(obs_params), length(observation));
for iPos = 1:length(obs_params)
    O(iPos, :) = pdf(obs_params{iPos}, observation);
end
% set likelihood to zero for impossible states
O(isnan(O)) = 0;
end

function O = RNN(activations, obs_params)
% Use re-current neural network activation as observation probability
    barGrid = length(obs_params);
    O = zeros(barGrid, size(activations, 1));
    O(1, :) = activations(:); % probability of being a beat position
    O(2:end, :) = repmat((1-activations(:)) / ...
        (barGrid-1), 1, barGrid-1)'; % probability of being a non-beat position
end