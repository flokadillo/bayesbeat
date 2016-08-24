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
if ~ismember(obj.dist_type, {'MOG', 'RNN', 'RNN_db'})
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
    if size(activations,2) > 1%this edition for AH@ISMIR2016 makes it possible to use the cnn acitvations also for beat tracking only.
        activations = activations(:,2);
    end
    O(1, :) = activations(:); % probability of being a beat position
    O(2:end, :) = repmat((1-activations(:)) / (barGrid-1), 1, barGrid-1)'; % probability of being a non-beat position
end

function O = RNN_db(activations, obs_params)
% AH@ISMIR2016
% Use re-current neural network activation as observation probability. It
% is assumed that there are two activations: one for beat, and one for the
% downbeat. In the experiments, slight improvement was obtained when taking
% into account the neighboring bins next to beats and downbeats as well.
    obs_params = cell2mat(obs_params);
    barGrid = length(obs_params);
    Od = zeros(barGrid, size(activations, 1));
    Ob = zeros(barGrid, size(activations, 1));
    beatactivations = activations(:,2);
    downbeatactivations = activations(:,1);
    Od(1, :) = downbeatactivations(:); % probability of being a beat position
    Od(2:end, :) = repmat((1-downbeatactivations(:)) / ...
        (barGrid-1), 1, barGrid-1)'; % probability of being a non-beat position
    numbeatpos = sum(obs_params==1);
    beatpos = find(obs_params==1);
    Ob(beatpos,:) = repmat(beatactivations(:).*(1-downbeatactivations(:)) ,1,numbeatpos)';%;/ (numbeatpos),1,numbeatpos)'; % probability of being a beat position   
    Ob(obs_params==0,:) = repmat((1-beatactivations(:)).*(1-downbeatactivations(:)) , 1, barGrid-numbeatpos)';%;/ (barGrid-numbeatpos), 1, barGrid-numbeatpos)'; % probability of being a non-beat position; 
    Ob(beatpos+1,:) = repmat(beatactivations(:).*(1-downbeatactivations(:)) ,1,numbeatpos)';%bin next to a beat position
    Ob(1, :) = downbeatactivations(:).*beatactivations(:); % probability of being a downbeat position
    Ob(2, :) = downbeatactivations(:).*beatactivations(:); % bin next to the downbeat
    Ob = Ob./repmat(sum(Ob),size(Ob,1),1);
    O = Ob;
end