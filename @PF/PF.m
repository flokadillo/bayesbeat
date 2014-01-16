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
        initial_r           % initial value of r for each particle
        nParticles
        particles
        sigma_N
        bin2decVector       % vector to compute indices for disc_trans_mat quickly
        ratio_Neff
        resampling_scheme   % type of resampling employed
        rbpf                % rbpf == 1, do rao-blackwellization on the discrete states
        warp_fun            % function that warps the weights w to a compressed space
        
    end
    
    methods
        function obj = PF(Params, rhythm2meter)
            addpath '~/diss/src/matlab/libs/bnt/KPMtools' % logsumexp
            addpath '~/diss/src/matlab/libs/pmtk3-1nov12/matlabTools/stats' % normalizeLogspace
            obj.M = Params.M;
            obj.Meff = Params.Meff;
            obj.N = Params.N;
            obj.R = Params.R;
            obj.T = Params.T;
            obj.nParticles = Params.nParticles;
            obj.sigma_N = Params.sigmaN;
            obj.pr = Params.pr;
            obj.pt = Params.pt;
            obj.barGrid = max(Params.barGrid_eff);
            obj.frame_length = Params.frame_length;
            obj.dist_type = Params.observationModelType;
            obj.rhythm2meter = rhythm2meter;
            obj.meter_state2meter = Params.meters;
            obj.ratio_Neff = Params.ratio_Neff;
            obj.rbpf = Params.rbpf;
            obj.resampling_scheme = Params.resampling_scheme;
            obj.warp_fun = Params.warp_fun;
            RandStream.setDefaultStream(RandStream('mt19937ar','seed',sum(100*clock)));
        end
        
        function obj = make_initial_distribution(obj, use_tempo_prior, tempo_per_cluster)
            obj.initial_m = zeros(obj.nParticles, 1);
            obj.initial_n = zeros(obj.nParticles, 1);
            obj.initial_r = zeros(obj.nParticles, 1);
            
            % use pseudo random monte carlo
            min_N = min(obj.minN);
            max_N = max(obj.maxN);
            n_grid = min_N:max_N;
            n_m_cells = floor(obj.nParticles / length(n_grid));
            
            
%             n_m_cells = floor(n_m_cells / (sum(obj.Meff)/obj.M));
            m_grid_size = sum(obj.Meff) / n_m_cells;
            r_m = rand(obj.nParticles, 1) - 0.5; % between -0.5 and +0.5
            r_n = rand(obj.nParticles, 1) - 0.5;
            c=1;
            for iR = 1:obj.R
                % create positions between 1 and obj.Meff(obj.rhythm2meter(iR))
                m_grid = 1+m_grid_size/2:m_grid_size:(obj.Meff(obj.rhythm2meter(iR))-m_grid_size/2);
                nParts(iR) = length(m_grid) * length(n_grid);
                temp = repmat(m_grid, length(n_grid), 1);
                obj.initial_m(c:c+nParts(iR)-1) = temp(:)+ r_m(c:c+nParts(iR)-1) * m_grid_size;
                obj.initial_n(c:c+nParts(iR)-1) = repmat(n_grid, 1, length(m_grid))' + r_n(c:c+nParts(iR)-1);
                obj.initial_r(c:c+nParts(iR)-1) = iR;
                c = c + nParts(iR);
            end
            
            if sum(nParts) < obj.nParticles
                obj.initial_r(c:end) = round(rand(obj.nParticles+1-c, 1)) + 1;
                obj.initial_n(c:end) = (r_n(c:end) + 0.5) * (max_N - min_N) + min_N;
                obj.initial_m(c:end) = (r_m(c:end) + 0.5) .* obj.Meff(obj.rhythm2meter(obj.initial_r(c:end)))';
            end
%             % n
%             obj.initial_n = betarnd(2.222, 3.908, obj.nParticles, 1);
%             obj.initial_n = obj.initial_n * (max(obj.maxN)-min(obj.minN)) + min(obj.minN);
%             % m
%             if obj.rbpf
%                 obj.initial_m = repmat(rand(1, obj.nParticles) .* (obj.M-1) + 1, obj.R, 1);
%                 for iR=1:obj.R
%                     % check if m is outside Meff and if yes, correct
%                     M_eff_iR = obj.Meff(obj.rhythm2meter(iR));
%                     ind = (obj.initial_m(iR, :) > M_eff_iR + 1);
%                     temp = mod(obj.initial_m(iR, ind) - 1, M_eff_iR) + 1;
%                     % shift by random amount of beats
%                     obj.initial_m(iR, ind) = temp + floor(rand(1, sum(ind)) * obj.meter_state2meter(1, iR))*M_eff_iR / obj.meter_state2meter(2, iR);
%                     obj.initial_m(iR, ind) = mod(obj.initial_m(iR, ind) - 1, M_eff_iR) + 1;
%                 end
%             else
%                 % assume unit distribution for r-state
%                 r_parts = floor(linspace(1, obj.nParticles, obj.R + 1));
%                 obj.initial_m = zeros(obj.nParticles, 1);
%                 obj.initial_r = zeros(obj.nParticles, 1);
%                 for iR=1:obj.R
%                     M_eff_iR = obj.Meff(obj.rhythm2meter(iR));
%                     ind = r_parts(iR):r_parts(iR+1);
%                     obj.initial_m(ind) = rand(length(ind), 1) .* (M_eff_iR-1) + 1;
%                     obj.initial_r(ind) = ones(length(ind), 1) * iR;
%                 end
%             end
        end
        
        function obj = make_transition_model(obj, minTempo, maxTempo)
            % convert from BPM into barpositions / audio frame
            meter_denom = obj.meter_state2meter(2, :);
            meter_denom = meter_denom(obj.rhythm2meter);
            
            %TODO for each cluster use different minN and maxN
            obj.minN = min(round(obj.M * obj.frame_length * minTempo ./ (meter_denom * 60)));
            obj.maxN = max(round(obj.M * obj.frame_length * maxTempo ./ (meter_denom * 60)));
            
            if obj.rbpf
                obj.sample_trans_fun = @(x, y) (obj.propagate_particles_rbpf(x, y, obj.nParticles, obj.sigma_N, ...
                    obj.minN, obj.maxN, obj.M, obj.Meff, obj.rhythm2meter));
                %             obj.sample_trans_fun = @(x, y) (x+y+obj.nParticles);
                obj = obj.createTransitionCPD;
            else
                obj.sample_trans_fun = @(x) obj.propagate_particles_pf(obj, x);
            end
        end
        
        function obj = make_observation_model(obj, data_file_pattern_barpos_dim)
            
            % Create observation model
            obj.obs_model = ObservationModel(obj.dist_type, obj.rhythm2meter, ...
                obj.meter_state2meter, obj.M, obj.N, obj.R, obj.barGrid, obj.Meff);
            
            % Train model
            obj.obs_model = obj.obs_model.train_model(data_file_pattern_barpos_dim);
            
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
        
        function [beats, tempo, rhythm, meter] = do_inference(obj, y, fname, inferenceMethod)
            
            % compute observation likelihoods
            obs_lik = obj.obs_model.compute_obs_lik(y);
            if obj.rbpf
                obj = obj.rbpf_apf(obs_lik, fname);
            else
                obj = obj.pf(obs_lik, fname);
            end
            [m_path, n_path, r_path] = obj.path_via_best_weight();
            % meter path
            t_path = obj.rhythm2meter(r_path);
            % compute beat times and bar positions of beats
            meter = obj.meter_state2meter(:, t_path);
            beats = obj.find_beat_times(m_path, t_path);
            tempo = meter(2, :)' .* 60 .* n_path / (obj.M * obj.frame_length);
            rhythm = r_path;
        end
    end
    
    methods (Access=protected)
        
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
        
        function lik = compute_obs_lik(obj, states_m_r, iFrame, obslik, m_per_grid)
            % states_m_r:   is a [nParts x 2] matrix, where (:, 1) are the
            %               m-values and (:, 2) are the r-values
            % obslik:       likelihood values [R, barGrid, nFrames]
            subind = floor((states_m_r(:, 1)-1) / m_per_grid) + 1;
            obslik = obslik(:, :, iFrame);
            %             r_ind = bsxfun(@times, (1:obj.R)', ones(1, obj.nParticles));
            ind = sub2ind([obj.R, obj.barGrid], states_m_r(:, 2), subind(:));
            %             lik = reshape(obslik(ind), obj.R, obj.nParticles);
            lik = obslik(ind);
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
        
        function [m_path, n_path, r_path] = path_via_best_weight(obj)
            % use particle with highest weight
            % ------------------------------------------------------------
            [~, bestParticle] = max(obj.particles.weight);
            
            if obj.rbpf
                % Backtracing:
                nFrames = size(obj.particles.n, 2);
                r_path = zeros(nFrames,1);
                [ ~, r_path(nFrames)] = max(obj.particles.delta(bestParticle, :));
                for i=nFrames-1:-1:1
                    r_path(i) = obj.particles.psi_mat(bestParticle, r_path(i+1), i+1);
                end
                m_path = squeeze(obj.particles.m(:, bestParticle, :));
                ind = sub2ind([obj.R, nFrames], r_path', (1:nFrames));
                m_path = m_path(ind)';
            else
                m_path = obj.particles.m(bestParticle, :);
                r_path = obj.particles.r(bestParticle, :);
            end
            n_path = obj.particles.n(bestParticle, :)';
            
            
            %             [ posteriorMAP ] = comp_posterior_of_sequence( [bestpath.position, bestpath.tempo, bestpath.meter], y, o1, [], params);
            %             fprintf('log post best weight: %.2f\n', posteriorMAP.sum);
        end
        
        function obj = pf(obj, obs_lik, fname)
            
            save_data = 1;
            
            nFrames = size(obs_lik, 3);
            % bin2dec conversion vector
            if save_data
                logP_data_pf = log(zeros(obj.nParticles, 5, nFrames, 'single'));
            end
            obj.particles = Particles(obj.nParticles, nFrames);
            iFrame = 1;
            obj.particles.m(:, iFrame) = obj.initial_m;
            obj.particles.n(:, iFrame) = obj.initial_n;
            obj.particles.r(:, iFrame) = obj.initial_r;
            eval_lik = @(x, y) obj.compute_obs_lik(x, y, obs_lik, obj.M / obj.barGrid);
            obs = eval_lik([obj.initial_m, obj.initial_r], iFrame);
            obj.particles.weight = log(obs / sum(obs));
            states = [obj.particles.m(:, iFrame), obj.particles.n(:, iFrame), obj.particles.r(:, iFrame)];
            state_dims = [obj.M; obj.N; obj.R];
            if obj.resampling_scheme > 1
                groups = obj.divide_into_fixed_cells(states, state_dims, 16);
            else
                groups = ones(obj.nParticles, 1);
            end
            if save_data
                % save particle data for visualizing
                % position
                logP_data_pf(:, 1, iFrame) = obj.particles.m(:, iFrame);
                % tempo
                logP_data_pf(:, 2, iFrame) = obj.particles.n(:, iFrame);
                % rhythm
                logP_data_pf(:, 3, iFrame) = obj.particles.r(:, iFrame);
                % weights
                logP_data_pf(:, 4, iFrame) = obj.particles.weight;
                logP_data_pf(:, 5, iFrame) = repmat(groups(:), 1, iFrame);
            end
            %             iFrame = 2;
            %             obj = obj.propagate_particles_pf(iFrame);
            %             [obj.particles.m(:, :, iFrame), obj.particles.n(:, iFrame)] = obj.sample_trans_fun(obj.particles.m(:, :, iFrame - 1), ...
            %                 obj.particles.n(:, iFrame - 1));
            %             obj = obj.propagate_particles(iFrame);
            %             obj.particles.psi_mat(:, :, iFrame) = repmat(1:obj.R, obj.nParticles, 1);
            
            resampling_frames = [];
            for iFrame=2:nFrames
                % transition from iFrame-1 to iFrame
                obj = obj.propagate_particles_pf(iFrame, 'm');
                
                % evaluate particle at iFrame-1
                obs = eval_lik([obj.particles.m(:, iFrame), obj.particles.r(:, iFrame)], iFrame);
                obj.particles.weight = obj.particles.weight(:) + log(obs);
                
                % Normalise importance weights
                % ------------------------------------------------------------
                [obj.particles.weight, ~] = normalizeLogspace(obj.particles.weight');
                %                 obj.particles.weight = obj.particles.weight / sum(obj.particles.weight);
                Neff = 1/sum(exp(obj.particles.weight).^2);
                % Resampling
                % ------------------------------------------------------------
                if (Neff < obj.ratio_Neff * obj.nParticles) && (iFrame < nFrames)
                    resampling_frames = [resampling_frames; iFrame];
                    fprintf('Resampling at Neff=%.3f (frame %i)\n', Neff, iFrame);
                    if obj.resampling_scheme == 0
                        newIdx = obj.resampleSystematic(exp(obj.particles.weight));
                        obj.particles.copyParticles(newIdx);
                    
                    elseif obj.resampling_scheme == 1
                        % warping:
                        w = exp(obj.particles.weight);
                        f = str2func(obj.warp_fun);
                        w_warped = f(w);                
                        newIdx = obj.resampleSystematic(w_warped);
                        obj.particles.copyParticles(newIdx);
                        w_fac = w ./ w_warped;
                        obj.particles.weight = log( w_fac(newIdx) / sum(w_fac(newIdx)) );
                    
                    elseif obj.resampling_scheme == 2
                        % k-means clustering
                        states = [obj.particles.m(:, iFrame), obj.particles.n(:, iFrame-1), obj.particles.r(:, iFrame)];
                        state_dims = [obj.M; obj.N; obj.R];
                        groups = obj.divide_into_clusters(states, state_dims, groups);
                        [newIdx, outWeights, groups] = obj.resample_in_groups(groups, obj.particles.weight);
                        obj.particles.copyParticles(newIdx);
                        obj.particles.weight = outWeights';
                        
                    elseif obj.resampling_scheme == 3
                        % apf and k-means
                        states = [obj.particles.m(:, iFrame), obj.particles.n(:, iFrame-1), obj.particles.r(:, iFrame)];
                        state_dims = [obj.M; obj.N; obj.R];
                        groups = obj.divide_into_clusters(states, state_dims, groups);
                        f = str2func(obj.warp_fun);
                        [newIdx, outWeights, groups] = obj.resample_in_groups(groups, obj.particles.weight, f);
                        obj.particles.copyParticles(newIdx);
                        obj.particles.weight = outWeights';
                    else
                        fprintf('WARNING: Unknown resampling scheme!\n');
                    end
                end
                
                % transition from iFrame-1 to iFrame
                obj = obj.propagate_particles_pf(iFrame, 'n');
                
                if save_data
                    % save particle data for visualizing
                    % position
                    logP_data_pf(:, 1, iFrame) = obj.particles.m(:, iFrame);
                    % tempo
                    logP_data_pf(:, 2, iFrame) = obj.particles.n(:, iFrame);
                    % rhythm
                    logP_data_pf(:, 3, iFrame) = obj.particles.r(:, iFrame);
                    % weights
                    logP_data_pf(:, 4, iFrame) = obj.particles.weight;
                    % groups
                    logP_data_pf(:, 5, iFrame) = groups;
                end
                
            end
            fprintf('      Average resampling interval: %.2f frames\n', mean(diff(resampling_frames)));
            if save_data
                save(['~/diss/src/matlab/beat_tracking/bayes_beat/temp/', fname, '_pf.mat'], ...
                    'logP_data_pf');
            end
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
        
        function obj = propagate_particles_pf(obj, new_frame, variable)
            % propagate particles by sampling from the transition prior
            % from new_frame-1 to new_frame
            % variable: is a string that says what to propagate ('m', 'n', 'r', 'mn')
            
            % update m
            if ~isempty(strfind(variable, 'm'))
                obj.particles.m(:, new_frame) = obj.particles.m(:, new_frame-1) + obj.particles.n(:, new_frame-1);
                %             ind = find(obj.particles.m(:, new_frame) > obj.Meff(obj.rhythm2meter(obj.particles.r(:, new_frame-1)))');
                %             obj.particles.m(:, :, new_frame) = bsxfun(@mod, temp - 1, obj.Meff(obj.rhythm2meter)') + 1;
                %             ind = find(sum(bsxfun(@gt, m, Meff(rhythm2meter(r))')));
                obj.particles.m(:, new_frame) = mod(obj.particles.m(:, new_frame) - 1, obj.Meff(obj.rhythm2meter(obj.particles.r(:, new_frame-1)))') + 1;
                
                % update r
                obj.particles.r(:, new_frame) = obj.particles.r(:, new_frame-1);
            end
            
            % update n
            if ~isempty(strfind(variable, 'n'))
                obj.particles.n(:, new_frame) = obj.particles.n(:, new_frame-1) + randn(obj.nParticles, 1) * obj.sigma_N * obj.M;
                obj.particles.n((obj.particles.n(:, new_frame) > obj.maxN), new_frame) = obj.maxN;
                obj.particles.n((obj.particles.n(:, new_frame) < obj.minN), new_frame) = obj.minN;
            end
            
            
        end
        
        
    end
    
    methods (Static)
        %         outIndex = systematicR(inIndex,wn);
        outIndex = resampleSystematic( w, n_samples );
                
        function [groups] = divide_into_fixed_cells(states, state_dims, nCells)
            % divide space into fixed cells
            % states: [nParticles x nStates]
            % state_dim: [nStates x 1]
            groups = zeros(size(states, 1), 1);
            n_r_bins = state_dims(3);
            n_n_bins = floor(sqrt(nCells/n_r_bins));
            n_m_bins = floor(nCells / (n_n_bins * n_r_bins));
            
            m_edges = linspace(1, state_dims(1) + 1, n_m_bins + 1);
            n_edges = linspace(0, state_dims(2) + 1, n_n_bins + 1);
            for iR=1:state_dims(3)
                ind = find(states(:, 3) == iR);
                [~, BIN_m] = histc(states(ind, 1), m_edges);
                [~, BIN_n] = histc(states(ind, 2), n_edges);
                for m = 1:n_m_bins
                    for n=1:n_n_bins
                        ind2 = intersect(ind(BIN_m==m), ind(BIN_n==n));
                        groups(ind2) = sub2ind([n_m_bins, n_n_bins, state_dims(3)], m, n, iR);
                    end
                end
            end
            if sum(groups==0) > 0
                error('group assignment failed\n')
            end
        end
        
        function [groups] = divide_into_clusters(states, state_dims, groups_old)
            % states: [nParticles x nStates]
            % state_dim: [nStates x 1]
            % groups_old: [nParticles x 1] group labels of the particles
            %               after last resampling step (used for initialisation)
            
            warning('off');
            group_ids = unique(groups_old);
            k = length(group_ids); % number of clusters
            
            % adjust the range of each state variable to make equally
            % important for the clustering
            [~, max_ind] = max(state_dims);
            % multiplication factor for each state variable to compute
            % distance
            lambda1 = 1;
            states(:, 1) = (states(:, 1)-1) * lambda1;
%             lambda2 = state_dims(1) / state_dims(2);
            lambda2 = 14;
            states(:, 2) = states(:, 2) * lambda2;
%             lambda3 = state_dims(1);
            lambda3 = 1440;
            states(:, 3) = (states(:, 3)-1) * lambda3 + 1;
%             for iDim = 1:length(state_dims)
%                 if iDim == max_ind, continue; end
%                 states(:, iDim) = (states(:, iDim)-1) * (state_dims(max_ind)-1) / (state_dims(iDim)-1) + 1;
%             end
            % compute centroid of clusters
            centroids = zeros(k, length(state_dims));
            for iCluster = 1:k
                centroids(iCluster, :) = mean(states(groups_old == group_ids(iCluster) , :));
            end
                       
            % do k-means clustering
            options = statset('MaxIter', 1);
            [groups, centroids, total_dist_per_cluster] = kmeans(states, k, 'replicates', 1, ...
                'start', centroids, 'emptyaction', 'drop', 'Distance', 'cityblock', 'options', options);
            
            % check if centroids are too close
            merging = 1;
            merged = 0;
            thr = 50; % if distance < thr: merge 
            while merging
                D = squareform(pdist(centroids, 'cityblock'), 'tomatrix');
                ind = (tril(D, 0) > 0);
                D(ind) = nan;
                D(logical(eye(size(centroids, 1)))) = nan;
                
                % find minimum distance
                [min_D, arg_min] = min(D(:));
                if min_D > thr,
                    merging = 0;
                else
                    [c1, c2] = ind2sub([size(D)], arg_min);
                    groups(groups==c2(1)) = c1(1);
%                     fprintf('   merging cluster %i + %i > %i\n', c1(1), c2(1), c1(1))
                    centroids = centroids([1:c2(1)-1, c2(1)+1:end], :);
                    if length(c1) == 1,  merging = 0;  end
                    merged = 1;
                end
            end
            
            if merged
                [groups, centroids, total_dist_per_cluster] = kmeans(states, [], 'replicates', 1, ...
                    'start', centroids, 'emptyaction', 'drop', 'Distance', 'cityblock');
            end
            
            % check if cluster spread is too high
            split = 0;
            n_parts_per_cluster = hist(groups, 1:size(centroids, 1));
            thr_spread = 80;
            separate_cl_idx = find((total_dist_per_cluster ./ n_parts_per_cluster') > thr_spread);
            for iCluster = 1:length(separate_cl_idx)
%                 fprintf('   splitting cluster %i\n', separate_cl_idx(iCluster));
                parts_idx = (groups == separate_cl_idx(iCluster));
                [gps, C] = kmeans(states(parts_idx, :), 2, 'replicates', 2, ...
                    'Distance', 'cityblock');
                gps(gps==1) = separate_cl_idx(iCluster);
                gps(gps==2) = max(groups) + 1;
                groups(parts_idx) = gps;
                centroids(separate_cl_idx(iCluster), :) = C(1, :);
                centroids = [centroids; C(2, :)];
                split = 1;
            end
            
            if split
                [groups, ~, ~] = kmeans(states, [], 'replicates', 1, 'start', centroids, 'emptyaction', 'drop', ...
                    'Distance', 'cityblock');
            end
%             warning('on');
            valid_groups = unique(groups);
            fprintf('    %i clusters; ', length(valid_groups));
%             for i=1:length(valid_groups)
%                 fprintf('%i.', valid_groups(i)); 
%             end
            fprintf('\n');
            
        end
        
        [outIndex, outWeights, groups] = resample_in_groups(groups, weights, warp_fun);
        
    end
    
    
    
end
