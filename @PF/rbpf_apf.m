function bestpath = rbpf_apf(obj, y)
% rao-blackwellized particle filter
% compute p(z, r | y), where z is computed using a PF and r can be computed
% analytically

% delta: max( p(x(t-1)=i, x(t)=j | y(1:t)) ) over x(t-1)

video = 1;
plotting = 0;

nFrames = length(y);

particle = struct('m', zeros(obj.R, obj.nParticles, nFrames), 'n' , zeros(obj.nParticles, nFrames), ...
    'weight', zeros(obj.nParticles, 1), 'posteriorTR', zeros(obj.nParticles, obj.R), ...
    'psi_mat', uint8(zeros(obj.nParticles, obj.R, nFrames)), ...
    'delta', zeros(obj.nParticles, obj.R), ...
    'log_trans', zeros(obj.nParticles, nFrames), 'log_obs', zeros(obj.R, obj.nParticles, nFrames), ...
    'oldWeight', zeros(obj.nParticles, 1));
particle.nFrames = nFrames;
minTempo = round(min(obj.minN) * obj.M * obj.frame_length / 240);

Meff = (obj.rhythm2meter + 2) * obj.M / 4;
% profile on

% Initialization
% beta distributed between 0 and 1
n = betarnd(2.222, 3.908, obj.nParticles, 1); 
% expand to [minN ... maxN]
n = n * max(obj.maxN) + min(obj.minN); 
particle.n(:, 1) = n * obj.frame_length * obj.M / (4 * 60);

%generate initial position. position of one particle should be in phase
particle.m(2, :, 1) = rand(obj.nParticles, 1) .* (obj.M-1) + 1;
% particles that fit into triple meter:
ind = (particle.m(2, :, 1) < Meff(1) + 1);
particle.m(1, ind, 1) = particle.m(2, ind, 1);
temp = mod(particle.m(2, ~ind, 1)-1, Meff(1)) + 1; 
% shift 1, 2, or 3 beats
particle.m(1, ~ind, 1) = temp + floor(rand(1, sum(~ind)) * 3) * obj.M / 4;

particle.log_obs(:, :, 1) = o1.obsProbFunc(y(1), o1, particle.m(:, :, 1));
particle.log_obs(:, :, 1) = log(particle.log_obs(:, :, 1));
particle.weight = sum(particle.log_obs(:, :, 1))';
particle.weight = particle.weight / sum(particle.weight);
% posterior: p(r_t | z_1:t, y_1:t)
particle.posteriorTR = ones(obj.nParticles, obj.R) / obj.R;
particle.delta = ones(obj.nParticles, obj.R) / obj.R;

% lowerBoundLik = 1 / (obj.nParticles * obj.R * 50);

% start with one correct particle at least
particle.n(1, 1) = obj.nStatesAnn(1);
particle.m(obj.RStatesAnn(1), 1, 1) = obj.mStatesAnn(1);
particle.posteriorTR(1, obj.RStatesAnn(1)) = 1;
particle.posteriorTR(1, [1:obj.RStatesAnn(1)-1, ...
            obj.RStatesAnn(1) + 1:obj.R]) = 0;

% make transition cpd
transCPD = createTransitionCPD(params, o1.RhytPat);

% profile on
tic;

if video, [hf, ha1, ha2, ha3, ha4, aviobj] = init_plot_PF; end


% initialize Viterbi
particle.delta = o1.obsProbFunc(y(1), o1, particle.m(:, :, 1))'; % [nDiscStates, nPart]

iFrame = 2;
predicted_m = bsxfun(@plus, particle.m(:, :, iFrame-1), particle.n(:, iFrame-1)');
particle.m(:, :, iFrame) = mod(predicted_m - 1, repmat(Meff, 1, obj.nParticles)) + 1;
particle.psi_mat(:, :, iFrame) = repmat(1:obj.R, obj.nParticles, 1);
% Tempo n_k: sample from prior p(nk|nk-1)
particle.n(:, iFrame) = particle.n(:, iFrame-1) + randn(obj.nParticles, 1) * obj.sigmaN;
%     particle.log_trans(:, iFrame) = log(normpdf(particle.n(:, iFrame), particle.n(:, iFrame-1), obj.sigmaN));
particle.n((particle.n(:, iFrame) > obj.N), iFrame) = obj.N;
particle.n((particle.n(:, iFrame) < minTempo), iFrame) = minTempo;

done = 0;

% bin2dec conversion vector
bin2decVector = 2.^(obj.R-1:-1:0)';
for iFrame=3:nFrames
    
    % p(y_n | x_n_1)
    likPerState = o1.obsProbFunc(y(iFrame-1), o1, particle.m(:, :, iFrame-1));
    particle.log_obs(:, :, iFrame-1) = log(likPerState);
%     minObs = min(likPerState(:));
%     maxObs = max(likPerState(:));
%     fprintf('%i: \t%.2f\t, max=%.2f , min=%.2f\n', iFrame, maxObs/minObs, maxObs, minObs);
    
    barCrossing = particle.m(:, :, iFrame-1) < particle.m(:, :, iFrame-2);
    barCrossing = barCrossing' * bin2decVector;
    
    particle = update_delta_and_psi(particle, obj.R, transCPD, params, ...
        barCrossing, likPerState, iFrame);
    
    % prediction: p(r_t | y_1:t-1, x_1:t) =
    % sum_r(t-1){ p(r_t | r_t-1, x_t-1, x_t) * p(r_t-1 | y_1:t-1, x_1:t-1) }
    transMatVect = cell2mat(transCPD(barCrossing+1));
    
    postReshaped = repmat(particle.posteriorTR', obj.R, 1);
    postReshaped = reshape(postReshaped, obj.R, []);
    
    % transition to t: sum over t-1
    % p(r_t | y_1:t-1, x_1:t) = 
    % sum_r(t-1){ p(r_t-1 | y_1:t-1, x_1:t-1) * p(r_t | r_t-1) }
    prediction = sum(transMatVect' .* postReshaped);
    prediction = reshape(prediction, obj.R, obj.nParticles);
    particle.oldWeight = particle.weight;
    
    % weight prediction by likelihood of the particle
    % compute p(r_t, y_t | x_1:t, y_1:t-1) =
    % p(r_t | y_1:t-1, x_1:t) * p(y_t | y_1:t-1, x_1:t, r_t)
    particle.posteriorTR = prediction .* likPerState;
    
    % weight = p(y_t| y_1:t-1, x_1:t) =
    % sum_r_t { p(r_t, y_t | x_1:t, y_1:t-1) } * p(y_t-1| y_1:t-2, x_1:t-1)
    particle.weight = particle.weight .* (sum(particle.posteriorTR))';
    % normalize to get valid pdf
    particle.posteriorTR = (particle.posteriorTR ./ ...
        repmat(sum(particle.posteriorTR), obj.R, 1))';
    
    % Normalise importance weights
    % ------------------------------------------------------------
    particle.weight = particle.weight / sum(particle.weight);
    Neff = 1/sum(particle.weight.^2);
    
    % Exact step (update)
    % ------------------------------------------------------------
    
    
    
    % Resampling
    % ------------------------------------------------------------
    if (Neff < 0.2 * obj.nParticles) && (iFrame < nFrames)
        fprintf('Resampling at Neff=%.3f (frame %i)\n', Neff, iFrame);
        newIdx = systematicR(1:obj.nParticles, particle.weight);
        particle = copyParticles(particle, newIdx);

%         if rand < exp(-0.001*iFrame)
%             particle = do_MH_move1(particle, iFrame, params, minTempo, y, o1);
%         end
        
%        % insert one particle at the true position
%        particle.n(1, 1:iFrame-1) = obj.nStatesAnn(1:iFrame-1);
%        particle.m(obj.annStateSequ(iFrame-1, 3), 1, 1:iFrame-1) = obj.mStatesAnn(1:iFrame-1);
%        particle.posteriorTR(1, obj.annStateSequ(iFrame-1, 3)) = 1;
%        particle.posteriorTR(1, [1:obj.annStateSequ(iFrame-1, 3)-1, ...
%            obj.annStateSequ(iFrame-1, 3) + 1:obj.R]) = 0;
    end
    
    % Sample from continuous variables p(m_k, n_k | m_k-1, n_k-1, t_m-1)
    % -----------------------------------------------------------
    % Tempo n_k: sample from prior p(nk|nk-1)
    particle.n(:, iFrame) = particle.n(:, iFrame-1) + randn(obj.nParticles, 1) * obj.sigmaN;
    particle.n((particle.n(:, iFrame) > obj.N), iFrame) = obj.N;
    particle.n((particle.n(:, iFrame) < minTempo), iFrame) = minTempo;
    
    particle.log_trans(:, iFrame) = log(normpdf(particle.n(:, iFrame)-particle.n(:, iFrame-1), ...
        0, obj.sigmaN));
    
    predicted_m = bsxfun(@plus, particle.m(:, :, iFrame-1), particle.n(:, iFrame - 1)');
    particle.m(:, :, iFrame) = mod(predicted_m - 1, repmat(Meff, 1, obj.nParticles)) + 1;
    
    m = particle.m(obj.RStatesAnn(iFrame), :, iFrame)';
    n = particle.n(:, iFrame);
    dist = sqrt((obj.mStatesAnn(iFrame) - m).^2 + ((obj.nStatesAnn(iFrame) - n)*50).^2);
    
%     fprintf('%i \n', round(min(dist)));
    if video
        obslik = o1.obsProbFunc(y(iFrame), o1, [1:Meff(1) ones(1, diff(Meff)); 1:Meff(2)]);
        obslik(1, (Meff(1)+1):end) = 0;
        obslik = obslik / sum(obslik(:));
        [aviobj, hf, ha1, ha2, ha3, ha4] = plot_PF(particle, o1, y, params, ...
            hf, ha1, ha2, ha3, ha4, iFrame, aviobj, Neff, obslik);
    end
    
    if (min(dist) > 50) && (~done)
        done = 1;
        fprintf('Lost track at frame %i\n', iFrame);
    end
    
end

% profile viewer

% find MAP filtering state:
% ------------------------------------------------------------

if ismember(obj.doParticleViterbi, [0, 2])
    % use particle with highest weight
    % ------------------------------------------------------------
    [~, bestParticle] = max(particle.weight);
    
    % Backtracing:
    path = zeros(nFrames,1);
    [ ~, path(nFrames)] = max(particle.delta(bestParticle, :));
    for i=nFrames-1:-1:1
        path(i) = particle.psi_mat(bestParticle, path(i+1), i+1);
    end
    [bestpath.meter, bestpath.rhythm] = ind2sub([obj.T obj.R], path);
    
    bestpath.tempo = particle.n(bestParticle, :)';
    bestpath.position = squeeze(particle.m(:, bestParticle, :));
    ind = sub2ind([obj.R, nFrames], bestpath.meter', (1:nFrames));
    bestpath.position = bestpath.position(ind)';
    [ posteriorMAP ] = comp_posterior_of_sequence( [bestpath.position, bestpath.tempo, bestpath.meter], y, o1, [], params);
    fprintf('log post best weight: %.2f\n', posteriorMAP.sum);
end

if obj.doParticleViterbi > 0
    % use Viterbi
    % ------------------------------------------------------------
    [bestpath.positionVit, bestpath.tempoVit, bestpath.rVit] = viterbiPF(particle, y, o1, transCPD, params);
    if plotting
        figure(1); hold on; plot(bestpath.positionVit, 'r', 'DisplayName','Viterbi');
    end
%     [ posteriorVit ] = comp_posterior_of_sequence( [bestpath.positionVit, bestpath.tempoVit, bestpath.rVit], y, o1, [], params);
    %     fprintf('posteriorVit = %.2f\n', posteriorVit.sum);
end

save('~/diss/src/matlab/beat_tracking/HMM/temp/parts.mat', 'bestpath', 'particle');
% % scatter(particle.m(:, :, 1))
% % profile viewer
% fprintf('Computation time: %.2f min\n',toc/60);
if video
    aviobj = close(aviobj);
    close(hf)
end
if ismember(obj.doParticleViterbi, [0, 2]) && plotting
    figure(1); hold on; plot(bestpath.position, 'DisplayName','Best weight');
end
if plotting
    figure(1); plot(obj.annStateSequ(:, 1), 'g', 'DisplayName','Groundtruth')
    legend show
end

end

function particle = do_MH_move1(particle, iFrame, params, minTempo, y, o1)

% compute log posterior for old particles
logPost = sum(sum(particle.log_obs(:, :, 1:iFrame-1), 3)) + sum(particle.log_trans(:, 1:iFrame-1), 2)';

% create candidates
particleNew = particle;
if y(iFrame).onsets > 0.2 % tempo move
    fprintf('tempo move: ');
    r= 0.5; % sample from [r, 1/r]
    lambda = rand(obj.nParticles, 1) *( 1 / r - r) + r;
    % multiply tempo of each particle with lambda
    particleNew.n = bsxfun(@times, particle.n, lambda); 
    % if n>N_max, do not move
    outOfBounds = max(particleNew.n(:, 1:iFrame-1), [], 2) > obj.N;
    particleNew.n(outOfBounds, :) = particle.n(outOfBounds, :);
    % if n<N_min, do not move
    outOfBounds = min(particleNew.n(:, 1:iFrame-1), [], 2) < minTempo;
    particleNew.n(outOfBounds, :) = particle.n(outOfBounds, :);
    % compute new bar position that results from tempo move
    temp = cumsum(particleNew.n(:, 1:iFrame-1), 2);
    for iR = 1: obj.R
        % add cumsum to initial state of particle
        particleNew.m(iR, :, 2:iFrame) = bsxfun(@plus, temp, particle.m(iR, :, 1)');
        % apply mod
        particleNew.m(iR, :, 2:iFrame) = mod(squeeze(particleNew.m(iR, :, 2:iFrame)) - 1, ...
            repmat(Meff(iR), obj.nParticles, iFrame-1)) + 1;
    end
    % compute prior probability p(x_(1:t))
    particleNew.log_trans(:, 2:iFrame-1) = log(normpdf(diff(particleNew.n(:, 1:iFrame-1), [], 2), ...
        0, obj.sigmaN));
else % position move
    fprintf('position move: ');
    % random shift between 0 and Meff-1
    shift = (rand(obj.nParticles, obj.R) .* repmat(Meff'-1, obj.nParticles, 1) + 1)';
    temp = repmat(Meff', obj.nParticles, 1);
    particleNew.m(:, :, 1:iFrame) = bsxfun(@plus, particle.m(:, :, 1:iFrame), shift);
    particleNew.m(:, :, 1:iFrame) = bsxfun(@mod, particleNew.m(:, :, 1:iFrame) - 1, temp') + 1;
end

% compute log posterior for candidates
% logObs = zeros(obj.R, obj.nParticles);
for i=1:iFrame-1
   particleNew.log_obs(:, :, i) = log(o1.obsProbFunc(y(i), o1, particleNew.m(:, :, i)));
end
logObs = sum(sum(particleNew.log_obs(:, :, 1:iFrame-1), 3)); % integrate out discrete states
logTrans = sum(particleNew.log_trans(:, 2:iFrame-1), 2)';
logPostNew = logObs + logTrans;

% metropolis hastings
u=rand(obj.nParticles, 1);

acceptance = exp(logPostNew-logPost);
% acceptance = logPostNew./logPost;
ind = (u <= acceptance');
fprintf('%.2f accepted\n', sum(ind)/length(ind));

particle.n(ind, 1:iFrame-1) = particleNew.n(ind, 1:iFrame-1);
particle.m(:, ind, 1:iFrame) = particleNew.m(:, ind, 1:iFrame);
end

function particle = do_MH_move2(particle, particleBeforeResampling, iFrame, params, minTempo, y, o1)

% METROPOLIS-HASTINGS STEP:
% ========================
movedParticles = particle;

% move tempo state and resulting position state
movedParticles.n(:, iFrame) = movedParticles.n(:, iFrame-1) + randn(obj.nParticles, 1) * 3 * obj.sigmaN;
movedParticles.n((movedParticles.n(:, iFrame) > obj.N), iFrame) = obj.N;
movedParticles.n((movedParticles.n(:, iFrame) < minTempo), iFrame) = minTempo;
movedParticles.m(:, iFrame) = mod(movedParticles.m(:, iFrame-1) + movedParticles.n(:, iFrame-1) - 1, obj.M ) + 1;

% posterior probability of moved particles
obsPerStateMoved = o1.obsProbFunc(y(iFrame), o1, movedParticles.m(:, iFrame));
obsPerStateMoved = sum(obsPerStateMoved ./ sum(obsPerStateMoved(:)));
distanceMat = bsxfun(@minus, particleBeforeResampling.n(:, iFrame - 1), movedParticles.n(:, iFrame)');
transProb = bsxfun(@times, normpdf(distanceMat, 0, obj.sigmaN), particleBeforeResampling.oldWeight');
transProb = sum(transProb); % sum out x_t-1
likPerStateMoved = obsPerStateMoved .* transProb;
likPerStateMoved = likPerStateMoved / sum(likPerStateMoved);

% posterior probability of not-moved particles
obsPerState = o1.obsProbFunc(y(iFrame), o1, particle.m(:, iFrame));
obsPerState = sum(obsPerState ./ sum(obsPerState(:)));
distanceMat = bsxfun(@minus, particleBeforeResampling.n(:, iFrame - 1), particle.n(:, iFrame)');
transProb = bsxfun(@times, normpdf(distanceMat, 0, obj.sigmaN), particleBeforeResampling.oldWeight');
transProb = sum(transProb); % sum out x_t-1
likPerState = obsPerState .* transProb;
likPerState = likPerState / sum(likPerState);

% acceptance probability
acceptance = likPerStateMoved./likPerState;
ind = (rand(obj.nParticles, 1) <= acceptance');
fprintf('%.2f accepted\n', sum(ind) / length(ind));

particle.n(ind, iFrame) = movedParticles.n(ind, iFrame);
particle.m(ind, iFrame) = movedParticles.m(ind, iFrame);
end

function [m] = unwrap_m(m, M)
mdif = diff(m);
ind = find(mdif < 0);
for i = 1:length(ind)
    m(ind(i)+1:end) = m(ind(i)+1:end) + M;
end
end

function particle = update_delta_and_psi(particle, obj.R, transCPD, ...
    params, barCrossing, likPerState, iFrame)
% probability of best state sequence that ends with state x(t) = j
%   delta(j) = max_i{ p( X(1:t)=[x(1:t-1), x(t)=j] | y(1:t) ) }
% best precursor state of j
%   psi(j) = arg max_i { p(X(t)=j | X(t-1)=i) * delta(i)}
deltaEnlarged = repmat(particle.delta', obj.R , 1);
deltaEnlarged = reshape(deltaEnlarged, obj.R, []);
transMatVect = cell2mat(transCPD(barCrossing+1));
%   delta(i, t-1) * p( X(t)=j | X(t-1)=i )
prediction2ts = deltaEnlarged .* transMatVect';
%   max_i { delta(i, t-1) * p( X(t)=j | X(t-1)=i ) }
[particle.delta, psi2] = max(prediction2ts);
particle.delta = reshape(particle.delta, obj.R, obj.nParticles )';
particle.psi_mat(:, :, iFrame) = reshape(psi2, obj.R, obj.nParticles )';
%   delta(j, t) = p(y(t) | X(t) = j) * max_i { delta(i, t-1) * p( X(t)=j | X(t-1)=i ) }
particle.delta = particle.delta .* likPerState';
% renormalize over discrete states
particle.delta = particle.delta ./ repmat(sum(particle.delta, 2), 1, 2);

end


function [hf, ha1, ha2, ha3, ha4, aviobj] = init_plot_PF
% close all
hf = figure;
ha1 = subplot(6, 1, [1 2]);
set(ha1, 'DrawMode', 'fast');
ha2 = subplot(6, 1, 3);
ha3 = subplot(6, 1, [4 5]);
set(ha3, 'DrawMode', 'fast');
ha4 = subplot(6, 1, 6);
aviobj = avifile('./temp/example.avi','compression','None', 'fps', 50);
set(hf, 'Position', [200 0 600 600])
end

function [aviobj, hf, ha1, ha2, ha3, ha4] = plot_PF(particle, o1, y, params, hf, ha1, ha2, ha3, ha4, iFrame, aviobj, Neff, obslik)
%     width = 3;

X = particle.m(:, :, iFrame)';
Y = repmat(particle.n(:, iFrame), 1, obj.T);

% compute weights for each position of the particles
weights = bsxfun(@times, particle.weight, particle.posteriorTR);

% map weights to color (discrete space between 0 and 64)
ind = linspace(0.01, 0, 64);
a = repmat(ind, obj.nParticles * obj.T, 1);
b = repmat(weights(:), 1, 64);
[~, ind] = min(abs(a-b), [], 2);
ind(ind > 64) = 64; ind(ind < 1) = 1;
ind(ind < 1) = 1;
%     ind = ind * 10;
col = autumn;

% METER 1
scatter(ha1, X(:, 1), Y(:, 1), 20, col(ind(1:obj.nParticles), :), 'filled');
if obj.annStateSequ(iFrame, 3) == 1
    hold(ha1, 'on');
    scatter(ha1, obj.mStatesAnn(iFrame), obj.nStatesAnn(iFrame), 30, 'o');
    hold(ha1, 'off');
end
xlim(ha1, [1 Meff(1)])
ylim(ha1, [1 obj.N])
stairs(ha2, obslik(1, :)');
xlim(ha2, [1 Meff(1)])
ylim(ha2, [0 0.005])
title(sprintf('Frame: %i', iFrame));

% METER 2
scatter(ha3, X(:, 2), Y(:, 2), 20, col(ind(obj.nParticles+1:end), :), 'filled');
if obj.annStateSequ(iFrame, 3) == 2
    hold(ha3, 'on');
    scatter(ha3, obj.mStatesAnn(iFrame), obj.nStatesAnn(iFrame), 30, 'o');
    hold(ha3, 'off');
end
xlim(ha3, [1 obj.M])
ylim(ha3, [1 obj.N])
stairs(ha4, obslik(2, :)');
xlim(ha4, [1 obj.M])
ylim(ha4, [0 0.005])
title(sprintf('Frame: %i', iFrame));



F = getframe(hf);
aviobj = addframe(aviobj, F);
end

function transCPD = createTransitionCPD(params,RhytPat)
% create 2^T transition matrices for discrete states
% each matrix is of size [R x R]
% [meter x rhythmic patterns x 2]
% e.g. meter 1 barcrossing = 1, meter 2 barcrossing 0 -> 1 0 -> 2 1

temp = zeros(obj.R, obj.R, 2); % [i x j x barcrossings]
p_meter_constant = 1-(obj.pt /  (obj.T-1));
p_rhythm_constant = 1-obj.pr;

for barCrossing = 1:2
    for iCluster1 = 1:obj.R
        for iCluster2 = 1:obj.R
            
            % check if meter change
            if RhytPat(iCluster1).meter == RhytPat(iCluster2).meter
                % check if rhythm pattern change
                if iCluster1 == iCluster2 % rhyth pattern the same
                    if barCrossing == 1 % no crossing, rhythm+meter stays constant
                        prob = 1;
                    else % both stay constant despite crossing
                        prob = p_rhythm_constant * p_meter_constant;
                    end
                else
                    if barCrossing == 1 % no crossing, meter constant, rhythm change
                        prob = 0;
                    else % crossing, meter constant, rhythm change
                        prob = p_meter_constant * obj.pr/obj.R;
                    end
                end
                
            else % meter change
                if barCrossing == 1 % no crossing, meter change
                    prob = 0;
                else % crossing, meter change
                    prob = obj.pt;
                end
            end
            temp(iCluster1, iCluster2, barCrossing) = prob;
        end
        % renormalize
        temp(iCluster1, :, barCrossing) = ...
            temp(iCluster1, :, barCrossing) / sum(temp(iCluster1, :, barCrossing));
    end
end

transCPD = cell(2^obj.T, 1);
transCPD(:) = {zeros(obj.R, obj.R)};

for iTransMat=1:length(transCPD)
    barCrossings = dec2bin(iTransMat - 1, obj.T) - '0';
    for iCluster=1:obj.R
        meterIdx = RhytPat(iCluster).meter - 2;
        transCPD{iTransMat}(meterIdx, :) = temp(meterIdx, :, barCrossings(meterIdx) + 1);
    end
end

end


function particle = copyParticles(particle, newIdx)
% copy particles according to newIdx after resampling
nParticles = size(particle.m, 2);
particle.m = particle.m(:, newIdx, :);
particle.n = particle.n(newIdx, :);
particle.posteriorTR = particle.posteriorTR(newIdx, :);
particle.weight = ones(nParticles, 1) / nParticles;
particle.delta = particle.delta(newIdx, :);
particle.psi_mat = particle.psi_mat(newIdx, :, :);
particle.log_obs = particle.log_obs(:, newIdx, :);
particle.log_trans = particle.log_trans(newIdx, :);

end

function particle = saveParticles(particle, newIdx)
% copy particles according to newIdx after resampling
particle.m = particle.m(:, newIdx, :);
particle.n = particle.n(newIdx, :);
particle.posteriorTR = particle.posteriorTR(newIdx, :);
particle.weight = ones(particle.nParticles, 1) / particle.nParticles;
particle.delta = particle.delta(newIdx, :);
particle.psi_mat = particle.psi_mat(newIdx, :, :);

end


