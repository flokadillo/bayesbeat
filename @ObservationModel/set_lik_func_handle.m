function functionHandle = set_lik_func_handle(obj)
% functionHandle = set_lik_func_handle(obj)
% creates a function handle to the function given by functionName
%
% ------------------------------------------------------------------------
%INPUT parameters:
% 
%
%OUTPUT parameters:
% functionHandle : function handles that refers to functionName
%
% 06.09.2012 by Florian Krebs
% ------------------------------------------------------------------------

functionHandle = str2func(obj.dist_type);
end

function O = RNN(activations, obs_params)
    barGrid = length(obs_params);
    O = zeros(barGrid, size(activations, 1));
    O(1, :) = activations(:);
    O(2:end, :) = repmat((1-activations(:)) / ...
        (barGrid-1), 1, barGrid-1)';
end

function O = gamma(observation, obs_params)
observation(observation<eps) = eps;
O = zeros(length(obs_params), length(observation));
for iPos = 1:length(obs_params)
    O(iPos, :) = pdf('gamma', observation, ...
        obs_params{iPos}(1), obs_params{iPos}(2));
end
% set likelihood to zero for impossible states
O(isnan(O)) = 0;
end

function O = invGauss(observation, obs_params)
observation(observation<eps) = eps;
O = zeros(length(obs_params), length(observation));
for iPos = 1:length(obs_params)
    O(iPos, :) = pdf('inversegaussian', observation, ...
        obs_params{iPos}(1), obs_params{iPos}(2));
end
% set likelihood to zero for impossible states
O(isnan(O)) = 0;
end

function O = MOG(observation, obs_params)
% obs_params.obs:  parameters of gamma distribution for each state [states x 2]
O = zeros(length(obs_params), length(observation));
for iPos = 1:length(obs_params)
    O(iPos, :) = pdf(obs_params{iPos}, observation);
end
% set likelihood to zero for impossible states
O(isnan(O)) = 0;
end

function O = MOG3(observation, obs_params)
% obs_params.obs:  parameters of gamma distribution for each state [states x 2]
O = zeros(length(obs_params), length(observation));
for iPos = 1:length(obs_params)
    O(iPos, :) = pdf(obs_params{iPos}, reshape([observation.onsets], ...
        length(observation(1).onsets), [])');
end
% set likelihood to zero for impossible states
O(isnan(O)) = 0;
end

function O = MOGAH(observation, obs_params)
% obs_params.obs:  parameters of gamma distribution for each state [states x 2]
O = zeros(length(obs_params), length(observation));
onsets = [observation.onsets];
for iPos = 1:length(obs_params)
    
    indZeros = (onsets == 0);
    
    O(iPos, ~indZeros) = pdf(obs_params{iPos}{2}, reshape(onsets(~indZeros), ...
        length(observation(1).onsets), [])');
    O(iPos, indZeros) = obs_params{iPos}{1};
end

% set likelihood to zero for impossible states
O(isnan(O)) = 0;
end

function O = prodGamma(observation, obs_params)
% obs_params.obs:  parameters of gamma distribution for each state [states x 2]
O = ones(size(obs_params.obs, 1), 1);
for iDim=1:obs_params.nDim
    O = O .* gampdf(observation.onsets(iDim), obs_params.obs(:, 1, iDim), obs_params.obs(:, 2, iDim));
end
% set likelihood to zero for impossible states
O(isnan(O)) = 0;
end

function O = meanGamma(observation, obs_params)
% obs_params.obs:  parameters of gamma distribution for each state [states x 2]
O = ones(size(obs_params.obs, 1), obs_params.nDim);
for iDim=1:obs_params.nDim
    O(:, iDim) = gampdf(observation.onsets(iDim), obs_params.obs(:, 1, iDim), obs_params.obs(:, 2, iDim));
end
% set likelihood to zero for impossible states
O(isnan(O)) = 0;
O = mean(O, 2);
end

function O = gammaPF(observation, obs_params, m)
% obs_params.obs:  parameters of gamma distribution for each state [states x 2]
% O: [meter x rhythm]
% map continuous bar position to discrete grid
% make m a row vector
% if size(m, 1) > size(m, 2), m = m'; end
nDiscStates = length(obs_params.RhytPat);
O = zeros(nDiscStates, size(m, 2));
barPosPerGrid =obs_params.M /obs_params.barGrid;
discreteBarPos = floor((m - 1)/barPosPerGrid) + 1;
for iDiscreteState = 1:nDiscStates
    pdfParams = [obs_params.RhytPat(iDiscreteState).barPos.params]';
    meterIdx = obs_params.RhytPat(iDiscreteState).meter-2;
    O(iDiscreteState, :) = gampdf(observation.onsets, pdfParams(discreteBarPos(meterIdx, :), 1), pdfParams(discreteBarPos(meterIdx, :), 2));
end
% constrain range:
% make min = 1
% thresh = 1000;
% O = O / min(O(:));
% if max(O(:)) > thresh
%    O = (O-1) * thresh / max(O(:));
%    O = O + 1;
% end
end


function O = fixed(observation, obs_params)
% obs_params.obs: likelihood of being a beat/non-beat
O = obs_params.likweight * observation.onsets .* obs_params.obs(:,1) + ((1-observation.onsets)/obs_params.likweight) .* obs_params.obs(:,2);
end

function O = multivariateHistogram(observation, obs_params)
% obs_params needs the following fields:
% M,N,T,obsbarpos3,obsbarpos4,centerbins3, centerbins3

[~,ind1]=min(abs(obs_params.centerbins-observation.onsets(1)));
[~,ind2]=min(abs(obs_params.centerbins-observation.onsets(2)));
O = obs_params.obs(:, ind1, ind2);
end

function O = histogram(observation, obs_params)
% obs_params needs the following fields:
% M,N,T,obsbarpos3,obsbarpos4,centerbins3, centerbins3

% % hist
%     [~, ind] = min(abs(obs_params.centerbins - observation.onsets));
%     O = obs_params.obs(:, ind);

% histc
ind = find(observation.onsets >= obs_params.centerbins);
O = obs_params.obs(:, ind(end));
end

function O = gauss0(observation, obs_params)
% obs_params.obs:  parameters of gamma distribution for each state [states x 2]
O = zeros(length(obs_params), length(observation));
for iPos = 1:length(obs_params)
    O(iPos, :) = pdf('Normal', observation, obs_params{iPos}(1), obs_params{iPos}(2));
end
% O = normpdf(observation, obs_params{iPos}, obs_params{iPos});
% % set likelihood to zero for impossible states
O(isnan(O)) = 0;
end

function O = gauss(observation, obs_params)
% obs_params.obs:  parameters of gamma distribution for each state [states x 2]
O = zeros(length(obs_params), length(observation));
for iPos = 1:length(obs_params)
    O(iPos, :) = pdf('Normal', observation, obs_params{iPos}(1), obs_params{iPos}(2));
end
% O = normpdf(observation, obs_params{iPos}, obs_params{iPos});
% % set likelihood to zero for impossible states
O(isnan(O)) = 0;
end

function O = multivariateGauss0(observation, obs_params)
% obs_params.obs:  parameters of gamma distribution for each state [states x 2]
O = normpdf(observation.onsets, zeros(obs_params.nDim, 1), obs_params.obs(:, 2, iDim));
% set likelihood to zero for impossible states
O(isnan(O)) = 0;
end

function O = bivariateGauss(observation, obs_params)
% obs_params needs the following fields:
% M,N,T,obsbarpos3,obsbarpos4,centerbins3, centerbins3
O = zeros(obs_params.M  * obs_params.N * obs_params.T * obs_params.R, 1);
for iMeter = 1:length(obs_params.obs)
    nPos = length(obs_params.obs{iMeter});
    for iPos = 1:nPos
        y(iMeter).lik(iPos) = pdf(obs_params.obs{iMeter}(iPos).params, observation.onsets);
    end
    % expand to all states O = [numstates x 1]
    bBorders = round(linspace(1, obs_params.M, nPos+1));
    temp = zeros(obs_params.M, 1);
    for iBorder = 2:length(bBorders)
        temp(bBorders(iBorder-1):bBorders(iBorder)) =  y(iMeter).lik(iBorder-1);
    end
    meterStartInd = sub2ind([obs_params.M obs_params.N obs_params.T obs_params.R], 1, 1, iMeter, 1);
    meterEndInd = sub2ind([obs_params.M obs_params.N obs_params.T obs_params.R], obs_params.M, obs_params.N, iMeter, 1);
    O(meterStartInd:meterEndInd) = repmat(temp, 1, obs_params.N);
    %        [position_path, tempo_path, meter_path, ~] = ind2sub([param.M,
    %        param.N, param.T, param.R], bestpath);
end
end

function O = gammaTempo(observation, obs_params)
% obs_params.obs:  parameters of gamma distribution for each state [states x 2]
% observations [timeframes x 2]: [flux tempo]
O = gampdf(observation.onsets, obs_params.obs(:,1), obs_params.obs(:,2));
% set likelihood to zero for impossible states
O(isnan(O)) = 0;
[outMat] = replicateVecAlong(obs_params.M, obs_params.N, obs_params.T, obs_params.R, 2, observation.tempo);
O = outMat .* O;
end
