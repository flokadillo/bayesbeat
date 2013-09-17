classdef PF
    % Hidden Markov Model Class
    properties (SetAccess=private)
        M                   % number of positions in a 4/4 bar
        Meff                % number of positions per meter
        N                   % number of tempo states
        R                   % number of rhythmic pattern states
        T                   % number of meters
        pn                  % probability of a switch in tempo
        pr                  % probability of a switch in rhythmic pattern
        pt                  % probability of a switch in meter
        rhythm2meter        % assigns each rhythmic pattern to a meter state (1, 2, ...)
        meter_state2meter   % specifies meter for each meter state (9/8, 8/8, 4/4)
        barGrid             % number of different observation model params per bar (e.g., 64)
        minN                % min tempo (n_min) for each rhythmic pattern
        maxN                % max tempo (n_max) for each rhythmic pattern
        frame_length        % audio frame length in [sec]
        dist_type           % type of parametric distribution
        sample_trans_fun    % transition model
        disc_trans_mat      % transition matrices for discrete states
        obs_model           % observation model
        initial_m           % initial value of m for each particle
        initial_n           % initial value of n for each particle
        nParticles
        particles
        sigma_N
        bin2decVector       % vector to compute indices for disc_trans_mat quickly
        ratio_Neff
        
    end
    
    methods
        function obj = PF(Params, rhythm2meter)
            obj.M = Params.M;
            obj.Meff = Params.Meff;
            obj.N = Params.N;
            obj.R = Params.R;
            obj.T = Params.T;
            obj.nParticles = Params.nParticles;
            obj.sigma_N = Params.sigmaN;
            obj.pr = Params.pr;
            obj.pt = Params.pt;
            obj.barGrid = Params.barGrid;
            obj.frame_length = Params.frame_length;
            obj.dist_type = Params.observationModelType;
            obj.rhythm2meter = rhythm2meter;
            obj.meter_state2meter = Params.meters;
            obj.ratio_Neff = Params.ratio_Neff;
        end
        
        function obj = make_transition_model(obj, minTempo, maxTempo)
            % convert from BPM into barpositions / audio frame
            meter_denom = obj.meter_state2meter(2, :);
            meter_denom = meter_denom(obj.rhythm2meter);
            
            %TODO for each cluster use different minN and maxN
            obj.minN = min(round(obj.M * obj.frame_length * minTempo ./ (meter_denom * 60)));
            obj.maxN = max(round(obj.M * obj.frame_length * maxTempo ./ (meter_denom * 60)));
            
            obj.sample_trans_fun = @(x, y) (obj.propagate_particles(x, y, obj.nParticles, obj.sigma_N, obj.minN, obj.maxN, obj.M));
            obj = obj.createTransitionCPD;
        end
        
        function lik = compute_obs_lik(obj, m, iFrame, obslik, m_per_grid)
            subind = floor((m-1) / m_per_grid) + 1;
            obslik = obslik(:, :, iFrame);
            r_ind = bsxfun(@times, (1:obj.R)', ones(1, obj.nParticles));
            ind = sub2ind([obj.R, obj.barGrid], r_ind(:), subind(:));
            lik = reshape(obslik(ind), obj.R, obj.nParticles);
        end
        
        function obj = make_observation_model(obj, data_file_pattern_barpos_dim)
            
            % Create observation model
            obj.obs_model = ObservationModel(obj.dist_type, obj.rhythm2meter, ...
                obj.meter_state2meter, obj.M, obj.N, obj.R, obj.barGrid, obj.Meff);
            
            % Train model
            obj.obs_model = obj.obs_model.train_model(data_file_pattern_barpos_dim);
            
        end
        
        function obj = make_initial_distribution(obj, use_tempo_prior, tempo_per_cluster)
            % n
            obj.initial_n = betarnd(2.222, 3.908, obj.nParticles, 1);
            obj.initial_n = obj.initial_n * max(obj.maxN) + min(obj.minN);
            % m
            obj.initial_m = repmat(rand(1, obj.nParticles) .* (obj.M-1) + 1, obj.R, 1);
            for iR=1:obj.R
                % check if m is outside Meff and if yes, correct
                M_eff_iR = obj.Meff(obj.rhythm2meter(iR));
                ind = (obj.initial_m(iR, :) > M_eff_iR + 1);
                temp = mod(obj.initial_m(iR, ind) - 1, M_eff_iR) + 1;
                % shift by random amount of beats
                obj.initial_m(iR, ind) = temp + floor(rand(1, sum(ind)) * obj.meter_state2meter(1, iR))*M_eff_iR / obj.meter_state2meter(2, iR);
                obj.initial_m(iR, ind) = mod(obj.initial_m(iR, ind) - 1, M_eff_iR) + 1;
            end
        end
        
        function obj = retrain_observation_model(obj, data_file_pattern_barpos_dim, pattern_id)
            %{
            Retrains the observation model for all states corresponding to <pattern_id>.

            :param data_file_pattern_barpos_dim: training data
            :param pattern_id: pattern to be retrained. can be a vector, too.
            :returns: the retrained hmm object

            %}
            obj.obs_model = obj.obs_model.retrain_model(data_file_pattern_barpos_dim, pattern_id);
            %             obj.obs_model.learned_params{pattern_id, :} = ...
            %                 obj.obs_model.fit_distribution(data_file_pattern_barpos_dim(:, pattern_id, :, :));
            
        end
        
        function [beats, tempo, rhythm, meter] = do_inference(obj, y, fname)
            
            % compute observation likelihoods
            obs_lik = obj.obs_model.compute_obs_lik(y);
            % decode MAP state sequence using Viterbi
            obj = obj.rbpf_apf(obs_lik, fname);
            % factorial HMM: mega states -> substates
            [m_path, n_path, r_path] = obj.path_via_best_weight();
            % meter path
            t_path = obj.rhythm2meter(r_path);
            % compute beat times and bar positions of beats
            meter = obj.meter_state2meter(:, t_path);
            beats = obj.find_beat_times(m_path, t_path);
            tempo = meter(2, :)' .* 60 .* n_path / (obj.M * obj.frame_length);
            rhythm = r_path;
        end
        
        function obj = createTransitionCPD(obj)
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
                        if obj.rhythm2meter(iCluster1) == obj.rhythm2meter(iCluster2)
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
            
            obj.disc_trans_mat = cell(2^obj.R, 1);
            obj.disc_trans_mat(:) = {zeros(obj.R, obj.R)};
            
            for iTransMat=1:length(obj.disc_trans_mat)
                barCrossings = dec2bin(iTransMat - 1, obj.R) - '0';
                for iCluster=1:obj.R
                    obj.disc_trans_mat{iTransMat}(iCluster, :) = ...
                        temp(iCluster, :, barCrossings(iCluster) + 1);
                end
            end
            obj.bin2decVector = 2.^(obj.R-1:-1:0)';
        end
        
        function [m_path, n_path, r_path] = path_via_best_weight(obj)
            % use particle with highest weight
            % ------------------------------------------------------------
            [~, bestParticle] = max(obj.particles.weight);
            
            % Backtracing:
            nFrames = size(obj.particles.n, 2);
            r_path = zeros(nFrames,1);
            [ ~, r_path(nFrames)] = max(obj.particles.delta(bestParticle, :));
            for i=nFrames-1:-1:1
                r_path(i) = obj.particles.psi_mat(bestParticle, r_path(i+1), i+1);
            end
            
            n_path = obj.particles.n(bestParticle, :)';
            m_path = squeeze(obj.particles.m(:, bestParticle, :));
            ind = sub2ind([obj.R, nFrames], r_path', (1:nFrames));
            m_path = m_path(ind)';
            %             [ posteriorMAP ] = comp_posterior_of_sequence( [bestpath.position, bestpath.tempo, bestpath.meter], y, o1, [], params);
            %             fprintf('log post best weight: %.2f\n', posteriorMAP.sum);
        end
    end
    
    methods (Access=protected)
        
        function obj = rbpf_apf(obj, obs_lik, fname)
            nFrames = size(obs_lik, 3);
            % bin2dec conversion vector
            logP_data_pf = log(zeros(obj.nParticles, obj.R, 3, nFrames, 'single'));
            obj.particles = Particles(obj.nParticles, obj.R, nFrames);
            iFrame = 1;
            obj.particles.m(:, :, iFrame) = obj.initial_m;
            obj.particles.n(:, iFrame) = obj.initial_n;
            eval_lik = @(m, iFrame) obj.compute_obs_lik(m, iFrame, obs_lik, obj.M / obj.barGrid);
            obs = eval_lik(obj.initial_m, iFrame);
            obj.particles.weight = (sum(obs) / sum(obs(:)))';
            % posterior: p(r_t | z_1:t, y_1:t)
            obj.particles.posterior_r = ones(obj.nParticles, obj.R) / obj.R;
            obj.particles.delta = obs'; % [nDiscStates, nPart]
            % save particle data for visualizing
            % position
            logP_data_pf(:, :, 1, iFrame) = obj.particles.m(:, :, iFrame)';
            % tempo
            logP_data_pf(:, :, 2, iFrame) = repmat(obj.particles.n(:, iFrame), 1, 2);
            % weights
            logP_data_pf(:, :, 3, iFrame) = log(bsxfun(@times, obj.particles.weight, obj.particles.posterior_r));
            
            iFrame = 2;
            obj = obj.propagate_particles(iFrame);
            obj.particles.psi_mat(:, :, iFrame) = repmat(1:obj.R, obj.nParticles, 1);
            
            for iFrame=3:nFrames
                obs = eval_lik(obj.particles.m(:, :, iFrame-1), iFrame-1);
                % check if barcrossing occured
                barCrossing = (obj.particles.m(:, :, iFrame-1) < obj.particles.m(:, :, iFrame-2))' * obj.bin2decVector;
                obj = obj.update_delta_and_psi(barCrossing, obs, iFrame);
                % ------------------------------------------------------------
                % prediction: p(r_t | y_1:t-1, x_1:t) =
                % sum_r(t-1){ p(r_t | r_t-1, x_t-1, x_t) * p(r_t-1 | y_1:t-1, x_1:t-1) }
                % p(r_t | r_t-1, x_t-1, x_t);
                transMatVect = cell2mat(obj.disc_trans_mat(barCrossing+1));
                % p(r_t-1 | y_1:t-1, x_1:t-1)
                postReshaped = reshape(repmat(obj.particles.posterior_r', obj.R, 1), obj.R, []);
                % transition to t: sum over t-1
                prediction = reshape(sum(transMatVect' .* postReshaped), obj.R, obj.nParticles);
                % weight prediction by likelihood of the particle
                % compute p(r_t, y_t | x_1:t, y_1:t-1) =
                % p(r_t | y_1:t-1, x_1:t) * p(y_t | y_1:t-1, x_1:t, r_t)
                obj.particles.posterior_r = (prediction .* obs)';
                % weight = p(y_t| y_1:t-1, x_1:t) =
                % sum_r_t { p(r_t, y_t | x_1:t, y_1:t-1) } * p(y_t-1| y_1:t-2, x_1:t-1)
                obj.particles.old_weight = obj.particles.weight;
                obj.particles.weight = obj.particles.weight .* (sum(obj.particles.posterior_r, 2));
                % normalize to get valid pdf
                obj.particles.posterior_r = obj.particles.posterior_r ./ ...
                    repmat(sum(obj.particles.posterior_r, 2), 1, obj.R);
                % Normalise importance weights
                % ------------------------------------------------------------
                obj.particles.weight = obj.particles.weight / sum(obj.particles.weight);
                Neff = 1/sum(obj.particles.weight.^2);
                % Resampling
                % ------------------------------------------------------------
                if (Neff < obj.ratio_Neff * obj.nParticles) && (iFrame < nFrames)
                    %fprintf('Resampling at Neff=%.3f (frame %i)\n', Neff, iFrame);
                    newIdx = obj.resampleSystematic(obj.particles.weight);
                    obj.particles = obj.particles.copyParticles(newIdx);
                end
                % save particle data for visualizing
                % position
                logP_data_pf(:, :, 1, iFrame-1) = obj.particles.m(:, :, iFrame-1)';
                % tempo
                logP_data_pf(:, :, 2, iFrame-1) = repmat(obj.particles.n(:, iFrame-1), 1, 2);
                % weights
                logP_data_pf(:, :, 3, iFrame-1) = log(bsxfun(@times, obj.particles.weight, obj.particles.posterior_r));
                
                % transition from iFrame-1 to iFrame
                obj = obj.propagate_particles(iFrame);
            end
            save(['~/diss/src/matlab/beat_tracking/bayes_beat/temp/', fname, '_pf.mat'], ...
                'logP_data_pf');
        end
        
        function obj = update_delta_and_psi(obj, barCrossing, obslik, iFrame)
            % probability of best state sequence that ends with state x(t) = j
            %   delta(j) = max_i{ p( X(1:t)=[x(1:t-1), x(t)=j] | y(1:t) ) }
            % best precursor state of j
            %   psi(j) = arg max_i { p(X(t)=j | X(t-1)=i) * delta(i)}
            deltaEnlarged = repmat(obj.particles.delta', obj.R , 1);
            deltaEnlarged = reshape(deltaEnlarged, obj.R, []);
            transMatVect = cell2mat(obj.disc_trans_mat(barCrossing + 1));
            %   delta(i, t-1) * p( X(t)=j | X(t-1)=i )
            prediction2ts = deltaEnlarged .* transMatVect';
            %   max_i { delta(i, t-1) * p( X(t)=j | X(t-1)=i ) }
            [obj.particles.delta, psi2] = max(prediction2ts);
            obj.particles.delta = reshape(obj.particles.delta, obj.R, obj.nParticles )';
            obj.particles.psi_mat(:, :, iFrame) = reshape(psi2, obj.R, obj.nParticles )';
            %   delta(j, t) = p(y(t) | X(t) = j) * max_i { delta(i, t-1) * p( X(t)=j | X(t-1)=i ) }
            obj.particles.delta = obj.particles.delta .* obslik';
            % renormalize over discrete states
            obj.particles.delta = obj.particles.delta ./ repmat(sum(obj.particles.delta, 2), 1, 2);
            
        end
        
        function obj = propagate_particles(obj, new_frame)
            % propagate particles by sampling from the transition prior
            % 'm', zeros(params.R, params.nParticles, nFrames)
            % 'n', zeros(params.nParticles, nFrames)
            obj.particles.n(:, new_frame) = obj.particles.n(:, new_frame - 1) + randn(obj.nParticles, 1) * obj.sigma_N * obj.M;
            obj.particles.n((obj.particles.n(:, new_frame) > obj.maxN), new_frame) = obj.maxN;
            obj.particles.n((obj.particles.n(:, new_frame) < obj.minN), new_frame) = obj.minN;
            temp = bsxfun(@plus, obj.particles.m(:, :, new_frame - 1), obj.particles.n(:, new_frame - 1)');
            %             obj.particles.m(:, :, new_frame) = bsxfun(@mod, temp - 1, obj.Meff(obj.rhythm2meter)') + 1;
            ind = find(sum(bsxfun(@gt, temp, obj.Meff(obj.rhythm2meter)')));
            temp(:, ind) = bsxfun(@mod, temp(:, ind) - 1, obj.Meff(obj.rhythm2meter)') + 1;
            %             obj.particles = obj.particles.update_m(temp, new_frame);
            % TODO: why does this step take so long ?
            obj.particles.m(:, :, new_frame) = temp;
        end
        
        function bestpath = pf(obj, obs_lik)
            nFrames = size(obs_lik, 3);
            obj.particles = Particles(obj.nParticles, nFrames, obj.R);
            eval_lik = @(m, iFrame) compute_obs_lik(m, iFrame, obs_lik, M_grid, obj.barGrid);
            particle.log_obs(:, :, 1) = o1.obsProbFunc(y(1), o1, particle.m(:, :, 1));
            particle.log_obs(:, :, 1) = log(particle.log_obs(:, :, 1));
            particle.weight = sum(particle.log_obs(:, :, 1))';
            particle.weight = particle.weight / sum(particle.weight);
            % posterior: p(r_t | z_1:t, y_1:t)
            particle.posteriorTR = ones(obj.nParticles, R) / R;
            particle.delta = ones(obj.nParticles, nDiscreteStates) / nDiscreteStates;
        end
        
        function beats = find_beat_times(obj, positionPath, meterPath)
            % [beats] = findBeatTimes(positionPath, meterPath, param_g)
            %   Find beat times from sequence of bar positions of the HMM beat tracker
            % ----------------------------------------------------------------------
            %INPUT parameter:
            % positionPath             : sequence of position states
            % meterPath                : sequence of meter states
            %                           NOTE: so far only median of sequence is used !
            % nBarPos                  : bar length in bar positions (=M)
            % framelength              : duration of being in one state in [sec]
            %
            %OUTPUT parameter:
            %
            % beats                    : [nBeats x 2] beat times in [sec] and
            %                           bar.beatnumber
            %
            % 29.7.2012 by Florian Krebs
            % ----------------------------------------------------------------------
            numframes = length(positionPath);
            meter = obj.meter_state2meter(:, meterPath);
            % TODO: implement for changes in meter
            if round(median(meter(1, :))) == 3 % 3/4
                numbeats = 3;
                denom = 4;
            elseif round(median(meter(1, :))) == 4 % 4/4
                numbeats = 4;
                denom = 4;
            elseif round(median(meter(1, :))) == 8 % 8/8
                numbeats = 8;
                denom = 8;
            elseif round(median(meter(1, :))) == 9 % 9/8
                numbeats = 9;
                denom = 8;
            else
                error('Meter %i not supported yet!\n', median(meterPath));
            end
            
            beatpositions =  round(linspace(1, obj.Meff(median(meterPath)), numbeats+1));
            beatpositions = beatpositions(1:end-1);
            %             beatpositions = [1; round(obj.M/4)+1; round(obj.M/2)+1; round(3*obj.M/4)+1];
            
            beats = [];
            beatno = [];
            beatco = 0;
            for i = 1:numframes-1
                for b = 1:numbeats
                    if positionPath(i) == beatpositions(b)
                        beats = [beats; i];
                        beatno = [beatno; beatco + b/10];
                        if b == numbeats, beatco = beatco + 1; end
                    elseif ((positionPath(i) > beatpositions(b)) && (positionPath(i+1) > beatpositions(b)) && (positionPath(i) > positionPath(i+1)))
                        % transition of two bars
                        bt = interp1([positionPath(i); obj.M+positionPath(i+1)],[i; i+1],obj.M+beatpositions(b));
                        beats = [beats; round(bt)];
                        beatno = [beatno; beatco + b/10];
                        if b == numbeats, beatco = beatco + 1; end
                    elseif ((positionPath(i) < beatpositions(b)) && (positionPath(i+1) > beatpositions(b)))
                        bt = interp1([positionPath(i); positionPath(i+1)],[i; i+1],beatpositions(b));
                        beats = [beats; round(bt)];
                        beatno = [beatno; beatco + b/10];
                        if b == numbeats, beatco = beatco + 1; end
                    end
                end
            end
            % if positionPath(i) == beatpositions(b), beats = [beats; i]; end
            
            beats = beats * obj.frame_length;
            beats = [beats beatno];
        end
        
    end
    
    methods (Static)
        %         outIndex = systematicR(inIndex,wn);
        outIndex = resampleSystematic( w );
        
    end
    
    
    
end
