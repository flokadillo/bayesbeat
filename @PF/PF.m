classdef PF < handle
    % Hidden Markov Model Class
    properties (SetAccess=private)
        M                           % number of positions in a 4/4 bar
        Meff                        % number of positions per meter
        N                           % number of tempo states
        R                           % number of rhythmic pattern states
        T                           % number of meters
        pn                          % probability of a switch in tempo
        pr                          % probability of a switch in rhythmic pattern
        rhythm2meter_state          % assigns each rhythmic pattern to a meter state (1, 2, ...)
        meter_state2meter           % specifies meter for each meter state (9/8, 8/8, 4/4)
        barGrid                     % number of different observation model params per bar (e.g., 64)
        minN                        % min tempo (n_min) for each rhythmic pattern
        maxN                        % max tempo (n_max) for each rhythmic pattern
        frame_length                % audio frame length in [sec]
        dist_type                   % type of parametric distribution
        sample_trans_fun            % transition model
        disc_trans_mat              % transition matrices for discrete states
        obs_model                   % observation model
        initial_m                   % initial value of m for each particle
        initial_n                   % initial value of n for each particle
        initial_r                   % initial value of r for each particle
        nParticles                  % number of particles
        particles
        particles_grid              % particle grid for viterbi
        sigma_N                     % standard deviation of tempo transition
        bin2decVector               % vector to compute indices for disc_trans_mat quickly
        ratio_Neff                  % reample if NESS < ratio_Neff * nParticles
        resampling_interval         % fixed resampling interval [samples]
        resampling_scheme           % type of resampling employed
        warp_fun                    % function that warps the weights w to a compressed space
        do_viterbi_filtering
        save_inference_data         % save intermediate output of particle filter for visualisation
        state_distance_coefficients % distance factors for k-means clustering [1 x nDims]
        cluster_merging_thr         % if distance < thr: merge
        cluster_splitting_thr       % if spread > thr: split
        inferenceMethod             % 'PF'
        n_max_clusters              % If number of clusters > n_max_clusters, kill cluster with lowest weight
        n_initial_clusters          % Number of cluster to start with
        rhythm_names                % cell array of rhythmic pattern names
        train_dataset                % dataset, which PF was trained on
        pattern_size                % size of one rhythmical pattern {'beat', 'bar'}
        use_silence_state           % state that describes non-musical content
    end
    
    methods
        function obj = PF(Params, rhythm2meter_state, rhythm_names)
            addpath '~/diss/src/matlab/libs/bnt/KPMtools' % logsumexp
            addpath '~/diss/src/matlab/libs/pmtk3-1nov12/matlabTools/stats' % normalizeLogspace
            obj.M = Params.M;
            obj.N = Params.N;
            obj.R = Params.R;
            obj.T = size(Params.meters, 2);
            obj.nParticles = Params.nParticles;
            obj.sigma_N = Params.sigmaN;
            obj.pr = Params.pr;
            obj.barGrid = max(Params.whole_note_div * (Params.meters(1, :) ./ Params.meters(2, :)));
            obj.frame_length = Params.frame_length;
            obj.dist_type = Params.observationModelType;
            obj.rhythm2meter_state = rhythm2meter_state;
            obj.meter_state2meter = Params.meters;
            obj.Meff = round((Params.meters(1, :) ./ Params.meters(2, :)) * (Params.M ./ max(Params.meters(1, :) ./ Params.meters(2, :))));
            obj.ratio_Neff = Params.ratio_Neff;
            obj.resampling_scheme = Params.resampling_scheme;
            obj.warp_fun = Params.warp_fun;
            obj.do_viterbi_filtering = Params.do_viterbi_filtering;
            obj.save_inference_data = Params.save_inference_data;
            obj.state_distance_coefficients = Params.state_distance_coefficients;
            obj.cluster_merging_thr = Params.cluster_merging_thr;
            obj.cluster_splitting_thr = Params.cluster_splitting_thr;
            obj.n_max_clusters = Params.n_max_clusters;
            obj.n_initial_clusters = Params.n_initial_clusters;
            obj.rhythm_names = rhythm_names;
            obj.pattern_size = Params.pattern_size;
            obj.resampling_interval = Params.res_int;
            obj.use_silence_state = Params.use_silence_state;
            rng('shuffle');
            %             % Matlab 2010
            %             RandStream.setDefaultStream(RandStream('mt19937ar','seed',sum(100*clock)));
        end
        
        function obj = make_initial_distribution(obj, tempo_per_cluster)
            
            % TODO: Implement prior initial distribution for tempo
            obj.initial_m = zeros(obj.nParticles, 1);
            obj.initial_n = zeros(obj.nParticles, 1);
            obj.initial_r = zeros(obj.nParticles, 1);
            
            % use pseudo random monte carlo
            n_grid = min(obj.minN):max(obj.maxN);
            n_m_cells = floor(obj.nParticles / length(n_grid));
            m_grid_size = sum(obj.Meff(obj.rhythm2meter_state)) / n_m_cells;
            r_m = rand(obj.nParticles, 1) - 0.5; % between -0.5 and +0.5
            r_n = rand(obj.nParticles, 1) - 0.5;
            nParts = zeros(obj.R, 1);
            c=1;
            for iR = 1:obj.R
                % create positions between 1 and obj.Meff(obj.rhythm2meter_state(iR))
                m_grid = 1+m_grid_size/2:m_grid_size:(obj.Meff(obj.rhythm2meter_state(iR))-m_grid_size/2);
                n_grid_iR = obj.minN(iR):obj.maxN(iR);
                nParts(iR) = length(m_grid) * length(n_grid_iR);
                temp = repmat(m_grid, length(n_grid_iR), 1);
                obj.initial_m(c:c+nParts(iR)-1) = temp(:)+ r_m(c:c+nParts(iR)-1) * m_grid_size;
                obj.initial_n(c:c+nParts(iR)-1) = repmat(n_grid_iR, 1, length(m_grid))' + r_n(c:c+nParts(iR)-1);
                obj.initial_r(c:c+nParts(iR)-1) = iR;
                c = c + nParts(iR);
            end
            if sum(nParts) < obj.nParticles
                obj.initial_r(c:end) = round(rand(obj.nParticles+1-c, 1)) * (obj.R-1) + 1;
                obj.initial_n(c:end) = (r_n(c:end) + 0.5)' .* (obj.maxN(obj.initial_r(c:end)) - obj.minN(obj.initial_r(c:end))) + obj.minN(obj.initial_r(c:end));
                obj.initial_m(c:end) = (r_m(c:end) + 0.5) .* (obj.Meff(obj.rhythm2meter_state(obj.initial_r(c:end)))-1)' + 1;
            end
        end
        
        function obj = make_transition_model(obj, minTempo, maxTempo)
            % convert from BPM into barpositions / audio frame
            meter_num = obj.meter_state2meter(1, obj.rhythm2meter_state);
            
            if strcmp(obj.pattern_size, 'bar')
                obj.minN = floor(obj.Meff(obj.rhythm2meter_state) .* obj.frame_length .* minTempo ./ (meter_num * 60));
                obj.maxN = ceil(obj.Meff(obj.rhythm2meter_state) .* obj.frame_length .* maxTempo ./ (meter_num * 60));
            else
                obj.minN = floor(obj.M * obj.frame_length * minTempo ./ 60);
                obj.maxN = ceil(obj.M * obj.frame_length * maxTempo ./ 60);
            end
            
            obj.sample_trans_fun = @(x) obj.propagate_particles_pf(obj, x);
        end
        
        function obj = make_observation_model(obj, data_file_pattern_barpos_dim, train_dataset)
            
            % Create observation model
            obj.obs_model = ObservationModel(obj.dist_type, obj.rhythm2meter_state, ...
                obj.meter_state2meter, obj.M, obj.N, obj.R, obj.barGrid, obj.Meff, obj.use_silence_state);
            
            % Train model
            obj.obs_model = obj.obs_model.train_model(data_file_pattern_barpos_dim);
            
            obj.train_dataset = train_dataset;
            
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
        
        function results = do_inference(obj, y, fname, inference_method)
            if isempty(strfind(inference_method, 'PF'))
               error('Inference method %s not compatible with PF model\n', inference_method);
            end
            % compute observation likelihoods
            obs_lik = obj.obs_model.compute_obs_lik(y);
            obj = obj.pf(obs_lik, fname);
            [m_path, n_path, r_path] = obj.path_via_best_weight();
            [ joint_prob_best ] = obj.compute_joint_of_sequence([m_path, n_path, r_path], obs_lik);
            fprintf('    Best weight log joint = %.1f\n', joint_prob_best.sum);
            if strfind(inference_method, 'viterbi')
                [m_path_v, n_path_v, r_path_v] = obj.path_via_viterbi(obs_lik);
                [ joint_prob_vit ] = obj.compute_joint_of_sequence([m_path_v, n_path_v, r_path_v], obs_lik);
                fprintf('    Viterbi log joint = %.1f\n', joint_prob_vit.sum);
                if joint_prob_vit.sum > joint_prob_best.sum
                    % use viterbi results
                    m_path = m_path_v;
                    n_path = n_path_v;
                    r_path = r_path_v;
                end                
            end
            % meter path
            t_path = obj.rhythm2meter_state(r_path);
            
            % compute beat times and bar positions of beats
            results{3} = obj.meter_state2meter(:, t_path);
            results{1} = obj.find_beat_times(m_path, t_path);
            results{2} = results{3}(2, :)' .* 60 .* n_path / (obj.M * obj.frame_length);
            results{4} = r_path;            
        end
        
        function [ joint_prob ] = compute_joint_of_sequence(obj, state_sequence, obs_lik)
            %obs_lik [R x barPos x nFrames]
            
            nFrames = size(obs_lik, 3);
            eval_lik = @(x, y) obj.compute_obs_lik(x, y, obs_lik, obj.M / obj.barGrid);
            
            joint_prob.obslik = zeros(nFrames, 1);
            
            n_diffs = diff(double(state_sequence(:, 2))); % nDiff(1) = s_s(2) - s_s(1)
            joint_prob.trans = log(normpdf(n_diffs/obj.M, 0, obj.sigma_N));
            
            
            for iFrame = 1:nFrames
                joint_prob.obslik(iFrame) = log(eval_lik([state_sequence(iFrame, 1), state_sequence(iFrame, 3)], iFrame));
            end
            joint_prob.sum = sum(joint_prob.trans) + sum(joint_prob.obslik);
        end
    end
    
    methods (Access=protected)
        
        
        function lik = compute_obs_lik(obj, states_m_r, iFrame, obslik, m_per_grid)
            % states_m_r:   is a [nParts x 2] matrix, where (:, 1) are the
            %               m-values and (:, 2) are the r-values
            % obslik:       likelihood values [R, barGrid, nFrames]
            subind = floor((states_m_r(:, 1)-1) / m_per_grid) + 1;
            obslik = obslik(:, :, iFrame);
            %             r_ind = bsxfun(@times, (1:obj.R)', ones(1, obj.nParticles));
            try
                ind = sub2ind([obj.R, obj.barGrid], states_m_r(:, 2), subind(:));
                lik = obslik(ind);
            catch exception
                fprintf('dimensions R=%i, barGrid=%i, states_m=%.2f - %.2f, subind = %i - %i, m_per_grid=%.2f\n', ...
                    obj.R, obj.barGrid, min(states_m_r(:, 1)), max(states_m_r(:, 1)), ...
                    min(subind), max(subind), m_per_grid);
                %                dimensions R=2, barGrid=1 x 1 x 1, states_m_r=9.224340e-01 - 1, subind = 1.430080e+03 - 2
                
            end
            %             lik = reshape(obslik(ind), obj.R, obj.nParticles);
        end
        
        function [m_path, n_path, r_path] = path_via_best_weight(obj)
            % use particle with highest weight
            % ------------------------------------------------------------
            [~, bestParticle] = max(obj.particles.weight);
            
            m_path = obj.particles.m(bestParticle, :);
            r_path = obj.particles.r(bestParticle, :);
            n_path = obj.particles.n(bestParticle, :)';
            m_path = m_path(:);
            n_path = n_path(:);
            r_path = r_path(:);
        end
        
        function [m_path, n_path, r_path] = path_via_viterbi(obj, obs_lik)
            % Find viterbi path through the particle grid. One problem is
            % that because of the deterministic transition between two
            % neighboring m-states, each particle has only one possible
            % successor. What we can do instead is only to use the m-states
            % of the grid and find the most probable path between those.
            % The n states are then computed after the m states have been
            % determined
            
            psi = zeros(obj.particles_grid.nParticles, obj.particles_grid.nFrames, 'uint16');
            eval_lik = @(x, y) obj.compute_obs_lik(x, y, obs_lik, obj.M / obj.barGrid);
            delta = log(eval_lik([obj.particles_grid.m(:, 1), obj.particles_grid.r(:, 1)], 1));
            
            for iFrame = 2:obj.particles_grid.nFrames
                
                % compute tempo matrix: rows are particles at iFrame-1,
                % cols are particles at iFrame
                tempo_current = bsxfun(@minus, obj.particles_grid.m(:, iFrame), obj.particles_grid.m(:, iFrame-1)')';
                rhythm_constant = bsxfun(@eq, obj.particles_grid.r(:, iFrame), obj.particles_grid.r(:, iFrame-1)')';
                % estimate for n_iFrame-1
                
                if iFrame == 2
                    tempo_prev = obj.particles_grid.n(:, 1);
                else
                    tempo_prev = obj.particles_grid.m(:, iFrame-1) - obj.particles_grid.m(psi(:, iFrame - 1 ), iFrame-2);
                    
                end
                % add Meff to negative tempi
                for iR=1:obj.R
                    rhythm_iR = bsxfun(@eq, obj.particles_grid.r(:, iFrame), (ones(obj.nParticles, 1)*iR)')';
                    idx = rhythm_constant & rhythm_iR & (tempo_current < 0);
                    tempo_current(idx) = tempo_current(idx) + obj.Meff(obj.rhythm2meter_state(iR));
                    idx = (obj.particles_grid.r(:, iFrame-1) == iR) & (tempo_prev < 0);
                    tempo_prev(idx) = tempo_prev(idx) + obj.Meff(obj.rhythm2meter_state(iR));
                end
                
                tempoChange = bsxfun(@minus, tempo_current, tempo_prev) / obj.M;
                logTransProbCont = log(zeros(size(tempoChange)));
                % to save computational power set probability of all transitions beyond 10
                % times std to zero
                transition_ok = abs(tempoChange) < (10 * obj.sigma_N);
                logTransProbCont(rhythm_constant & transition_ok) = ...
                    log(normpdf(tempoChange(rhythm_constant & transition_ok), 0, obj.sigma_N));
                
                % find best precursor particle
                [delta, psi(:, iFrame)] = max(bsxfun(@plus, logTransProbCont, delta(:)));
                delta = delta' + log(eval_lik([obj.particles_grid.m(:, iFrame), obj.particles_grid.r(:, iFrame)], iFrame));
            end
            
            % Termination
            particleTraj = zeros(obj.particles_grid.nFrames, 1);
            [logPost, particleTraj(end)] = max(delta(:));
            
            %             fprintf('    logPost(Viterbi) = %.2f\n', logPost);
            
            % Backtracking
            m_path = zeros(obj.particles_grid.nFrames, 1);
            n_path = zeros(obj.particles_grid.nFrames, 1);
            r_path = zeros(obj.particles_grid.nFrames, 1);
            m_path(end) = obj.particles_grid.m(particleTraj(end), obj.particles_grid.nFrames);
            n_path(end) = obj.particles_grid.n(particleTraj(end), obj.particles_grid.nFrames);
            r_path(end) = obj.particles_grid.r(particleTraj(end), obj.particles_grid.nFrames);
            
            for iFrame = obj.particles_grid.nFrames-1:-1:1
                particleTraj(iFrame) = psi(particleTraj(iFrame+1), iFrame+1);
                m_path(iFrame) = obj.particles_grid.m(particleTraj(iFrame), iFrame);
                n_path(iFrame) = obj.particles_grid.n(particleTraj(iFrame), iFrame);
                r_path(iFrame) = obj.particles_grid.r(particleTraj(iFrame), iFrame);
            end
            n_path = diff(m_path);
            n_path(n_path<0) = n_path(n_path<0) + obj.Meff(obj.rhythm2meter_state(r_path(n_path<0)))';
            n_path = [n_path; n_path(end)];
            
            m_path = m_path(:);
            n_path = n_path(:);
            r_path = r_path(:);
        end
        
        function obj = pf(obj, obs_lik, fname)
            addpath '~/diss/src/matlab/libs/bnt/KPMtools' % logsumexp
            addpath '~/diss/src/matlab/libs/pmtk3-1nov12/matlabTools/stats' % normalizeLogspace
            
            nFrames = size(obs_lik, 3);
            if obj.save_inference_data
                logP_data_pf = log(zeros(obj.nParticles, 5, nFrames, 'single'));
            end
            % initialize particles
            iFrame = 1;
            m = zeros(obj.nParticles, nFrames, 'single');
            n = zeros(obj.nParticles, nFrames, 'single');
            r = zeros(obj.nParticles, nFrames, 'single');
            m(:, iFrame) = obj.initial_m;
            n(:, iFrame) = obj.initial_n;
            r(:, iFrame) = obj.initial_r;
            
            if strcmp(obj.inferenceMethod, 'PF_viterbi')
                obj.particles_grid = Particles(obj.nParticles, nFrames);
                obj.particles_grid.m(:, iFrame) = obj.initial_m;
                obj.particles_grid.n(:, iFrame) = obj.initial_n;
                obj.particles_grid.r(:, iFrame) = obj.initial_r;
            end
            % observation probability
            eval_lik = @(x, y) obj.compute_obs_lik(x, y, obs_lik, obj.M / obj.barGrid);
            obs = eval_lik([obj.initial_m, obj.initial_r], iFrame);
            weight = log(obs / sum(obs));
            if obj.resampling_scheme > 1
                % divide particles into clusters by kmeans
                groups = obj.divide_into_fixed_cells([m(:, iFrame), n(:, iFrame), r(:, iFrame)], [obj.M; obj.N; obj.R], obj.n_initial_clusters);
                n_clusters = zeros(nFrames, 1);
            else
                groups = ones(obj.nParticles, 1);
            end
            if obj.save_inference_data
                % save particle data for visualizing
                logP_data_pf(:, 1, iFrame) = m(:, iFrame);
                logP_data_pf(:, 2, iFrame) = n(:, iFrame);
                logP_data_pf(:, 3, iFrame) = r(:, iFrame);
                logP_data_pf(:, 4, iFrame) = weight;
                logP_data_pf(:, 5, iFrame) = groups;
            end
            
            resampling_frames = zeros(nFrames, 1);

            for iFrame=2:nFrames
                % transition from iFrame-1 to iFrame
                %                 obj = obj.propagate_particles_pf(iFrame, 'm');
                m(:, iFrame) = m(:, iFrame-1) + n(:, iFrame-1);
                m(:, iFrame) = mod(m(:, iFrame) - 1, obj.Meff(obj.rhythm2meter_state(r(:, iFrame-1)))') + 1;
                r(:, iFrame) = r(:, iFrame-1);
                
                if obj.do_viterbi_filtering
                    [m, n, r, weight] = obj.viterbi_filtering(m, n, r, weight);
                end
                
                % evaluate particle at iFrame-1
                obs = eval_lik([m(:, iFrame), r(:, iFrame)], iFrame);
                weight = weight(:) + log(obs);

                % Normalise importance weights
                % ------------------------------------------------------------
                [weight, ~] = normalizeLogspace(weight');
                
                % Resampling
                % ------------------------------------------------------------
                if strcmp(obj.inferenceMethod, 'PF_viterbi')
                    n_last_step = n(:, iFrame-1);
                end
                if obj.resampling_interval == 0
                    Neff = 1/sum(exp(weight).^2);
                    do_resampling = (Neff < obj.ratio_Neff * obj.nParticles);
                else
                    do_resampling = (rem(iFrame, obj.resampling_interval) == 0);
                end
                
                if do_resampling && (iFrame < nFrames)
%                     fprintf('    Resampling at Neff=%.3f (frame %i)\n', Neff, iFrame);
                    resampling_frames(iFrame) = iFrame;
                    if obj.resampling_scheme == 2 || obj.resampling_scheme == 3 % MPF or AMPF
                        groups = obj.divide_into_clusters([m(:, iFrame), n(:, iFrame-1), r(:, iFrame)], [obj.M; obj.N; obj.R], groups);
                        n_clusters(iFrame) = length(unique(groups));
                    end
                    [weight, groups, newIdx] = obj.resample(weight, groups);
                    m(:, 1:iFrame) = m(newIdx, 1:iFrame);
                    r(:, 1:iFrame) = r(newIdx, 1:iFrame);
                    n(:, 1:iFrame) = n(newIdx, 1:iFrame);
                end
                
                % transition from iFrame-1 to iFrame
                %                 obj = obj.propagate_particles_pf(iFrame, 'n');
                n(:, iFrame) = n(:, iFrame-1) + randn(obj.nParticles, 1) * obj.sigma_N * obj.M;
                out_of_range = n(:, iFrame)' > obj.maxN(r(:, iFrame));
                n(out_of_range, iFrame) = obj.maxN(r(out_of_range, iFrame));
                out_of_range = n(:, iFrame)' < obj.minN(r(:, iFrame));
                n(out_of_range, iFrame) = obj.minN(r(out_of_range, iFrame));
               
                if strcmp(obj.inferenceMethod, 'PF_viterbi')
                    obj.particles_grid.m(:, iFrame) =  obj.particles.m(:, iFrame);
                    obj.particles_grid.n(:, iFrame) =  obj.particles.n(:, iFrame);
                    obj.particles_grid.r(:, iFrame) =  obj.particles.r(:, iFrame);
                end
                
                if obj.save_inference_data
                    % save particle data for visualizing
                    logP_data_pf(:, 1, iFrame) = m(:, iFrame);
                    logP_data_pf(:, 2, iFrame) = n(:, iFrame);
                    logP_data_pf(:, 3, iFrame) = r(:, iFrame);
                    logP_data_pf(:, 4, iFrame) = weight;
                    logP_data_pf(:, 5, iFrame) = groups;
                end
                
            end
            
            obj.particles.m = m;
            obj.particles.n = n;
            obj.particles.r = r;
            obj.particles.weight = weight;

            fprintf('    Average resampling interval: %.2f frames\n', mean(diff(resampling_frames(resampling_frames>0))));
            if obj.resampling_scheme > 1
                fprintf('    Average number of clusters: %.2f frames\n', mean(n_clusters(n_clusters>0)));
            end
            if obj.save_inference_data
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
                %             ind = find(obj.particles.m(:, new_frame) > obj.Meff(obj.rhythm2meter_state(obj.particles.r(:, new_frame-1)))');
                %             obj.particles.m(:, :, new_frame) = bsxfun(@mod, temp - 1, obj.Meff(obj.rhythm2meter_state)') + 1;
                %             ind = find(sum(bsxfun(@gt, m, Meff(rhythm2meter_state(r))')));
                obj.particles.m(:, new_frame) = mod(obj.particles.m(:, new_frame) - 1, obj.Meff(obj.rhythm2meter_state(obj.particles.r(:, new_frame-1)))') + 1;
                
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
        
        function [groups] = divide_into_clusters(obj, states, state_dims, groups_old)
            % states: [nParticles x nStates]
            % state_dim: [nStates x 1]
            % groups_old: [nParticles x 1] group labels of the particles
            %               after last resampling step (used for initialisation)
            %             addpath('/home/florian/diss/src/matlab/libs/fast_kmeans')
            %             addpath('/home/florian/diss/src/matlab/libs/litekmeans')
           warning('off');
            [group_ids, ~, IC] = unique(groups_old);
            %             fprintf('    %i groups >', length(group_ids));
            k = length(group_ids); % number of clusters
            
            % adjust the range of each state variable to make equally
            % important for the clustering
            points = zeros(obj.nParticles, length(state_dims)+1);
            points(:, 1) = (sin(states(:, 1)' * 2 * pi ./ obj.Meff(obj.rhythm2meter_state(states(:, 3)))) + 1) * obj.state_distance_coefficients(1);
            points(:, 2) = (cos(states(:, 1)' * 2 * pi ./ obj.Meff(obj.rhythm2meter_state(states(:, 3)))) + 1) * obj.state_distance_coefficients(1);
            points(:, 3) = states(:, 2) * obj.state_distance_coefficients(2);
            points(:, 4) =(states(:, 3)-1) * obj.state_distance_coefficients(3) + 1;
            
            % compute centroid of clusters
            % TODO: vectorise!
            centroids = zeros(k, length(state_dims)+1);
            for i_dim=1:size(points, 2)
                %                 centroids(iCluster, :) = mean(points(groups_old == group_ids(iCluster) , :));
                centroids(:, i_dim) = accumarray(IC, points(:, i_dim), [], @mean);
            end
            %             centroids_old = centroids;
            % K-MEANS CLUSTERING
            % ----------------------------------------------------------
            %             c = centroids.^2; % k x d
            %             x = points.^2; % n x d
            %             sq_diff = zeros(k, obj.nParticles);
            %             for i_dim=1:size(points, 2)
            %                 sq_diff = sq_diff + bsxfun(@minus, centroids(:, i_dim), points(:, i_dim)').^2;
            %             end
            %             [vals, groups] = min(sq_diff);
            %             [group_ids, ~, IC] = unique(groups);
            %             valid_groups = ismember(1:k, group_ids);
            %             mean_dist_per_cluster = accumarray(groups', vals, [], @mean);
            %             mean_dist_per_cluster = mean_dist_per_cluster(valid_groups);
            %
            %             centroids = zeros(length(group_ids), size(points, 2));
            %             for i_dim=1:size(points, 2)
            %                 centroids(:, i_dim) = accumarray(IC, points(:, i_dim), [], @mean);
            %             end
            % ----------------------------------------------------------
            %             col = hsv(64);
            %             rhyt_idx = states(:, 3)==2;
            %             fac = max([1, floor(64 / max(groups_old(rhyt_idx)))-1]);
            %             figure(1); scatter(states(rhyt_idx, 1), states(rhyt_idx, 2), [], col(groups_old(rhyt_idx) * fac, :), 'filled');
            
            % do k-means clustering
            options = statset('MaxIter', 1);
            [groups, centroids, total_dist_per_cluster] = kmeans(points, k, 'replicates', 1, ...
                'start', centroids, 'emptyaction', 'drop', 'Distance', 'sqEuclidean', 'options', options);
            %
            %             [groups, centroids, total_dist_per_cluster] = fast_kmeans(points', centroids', 1);
            
            %             centroids = litekmeans(X, centroids);
            
            %             remove empty clusters
            total_dist_per_cluster = total_dist_per_cluster(~isnan(centroids(:, 1)));
            centroids = centroids(~isnan(centroids(:, 1)), :);
            [group_ids, ~, j] = unique(groups);
            group_ids = 1:length(group_ids);
            groups = group_ids(j)';
            %                         figure(1); scatter(states(rhyt_idx, 1), states(rhyt_idx, 2), [], col(groups(rhyt_idx) * fac), 'filled');
            
            % check if centroids are too close
            %             fprintf('    merging %i > ', size(centroids, 1));
            merging = 1;
            merged = 0;
            while merging
                D = squareform(pdist(centroids, 'euclidean'), 'tomatrix');
                ind = (tril(D, 0) > 0);
                D(ind) = nan;
                D(logical(eye(size(centroids, 1)))) = nan;
                
                % find minimum distance
                [min_D, arg_min] = min(D(:));
                if min_D > obj.cluster_merging_thr,
                    merging = 0;
                else
                    [c1, c2] = ind2sub(size(D), arg_min);
                    groups(groups==c2(1)) = c1(1);
                    % new centroid
                    centroids(c1(1), :) = mean(points(groups==c1(1), :));
                    % squared Euclidean distance
                    total_dist_per_cluster(c1(1)) = sum(sum(bsxfun(@minus, points(groups==c1(1), :), centroids(c1(1), :)).^2));
                    %                     mean_dist_per_cluster(c1(1)) = mean(sum(bsxfun(@minus, points(groups==c1(1), :), centroids(c1(1), :)).^2));
                    if length(c1) == 1,  merging = 0;  end
                    % remove old centroid
                    centroids = centroids([1:c2(1)-1, c2(1)+1:end], :);
                    total_dist_per_cluster = total_dist_per_cluster([1:c2(1)-1, c2(1)+1:end]);
                    %                     mean_dist_per_cluster = mean_dist_per_cluster([1:c2(1)-1, c2(1)+1:end]);
                    merged = 1;
                end
            end
            %             fprintf('%i\n', size(centroids, 1));
            %             if merged
            %                 [groups, centroids, total_dist_per_cluster] = kmeans(states, [], 'replicates', 1, ...
            %                     'start', centroids, 'emptyaction', 'drop', 'Distance', 'cityblock', 'options', options);
            %             end
            
            % check if cluster spread is too high
            %             fprintf('    splitting %i > ', size(centroids, 1));
            split = 0;
            [group_ids, ~, j] = unique(groups);
            group_ids = 1:length(group_ids);
            groups = group_ids(j)';
            n_parts_per_cluster = hist(groups, 1:length(group_ids));
            separate_cl_idx = find((total_dist_per_cluster ./ n_parts_per_cluster') > obj.cluster_splitting_thr);
            %             separate_cl_idx = find(mean_dist_per_cluster > obj.cluster_splitting_thr);
            for iCluster = 1:length(separate_cl_idx)
                % find particles that belong to the cluster to split
                parts_idx = find((groups == separate_cl_idx(iCluster)));
                % put second half into a new group
                groups(parts_idx(round(length(parts_idx)/2)+1:end)) = max(groups) + 1;
                % update centroid
                centroids(separate_cl_idx(iCluster), :) = mean(points(parts_idx(1:round(length(parts_idx)/2)), :), 1);
                % add new centroid
                centroids = [centroids; mean(points(parts_idx(round(length(parts_idx)/2)+1:end), :), 1)];
                %                     try
                %                         centroids = [centroids; mean(points(parts_idx(round(length(parts_idx)/2)+1:end), :))];
                %                     catch exception
                %                         fprintf('Size centroids: %i x %i\n', size(centroids, 1), size(centroids, 2));
                %                         fprintf('length(parts_idx) = %i\n', length(parts_idx));
                %                         fprintf('mean(points): %i x %i\n', size(mean(points(parts_idx(1:round(length(parts_idx)/2)))), 1), size(mean(points(parts_idx(1:round(length(parts_idx)/2)))), 2));
                %                         error('Problem!\n');
                %                     end
                %                 end
                split = 1;
            end
            %             fprintf('%i\n', size(centroids, 1));
            
            if split || merged
                try
                    [groups, ~, ~] = kmeans(points, [], 'replicates', 1, 'start', centroids, 'emptyaction', 'drop', ...
                        'Distance', 'sqEuclidean', 'options', options);
                catch exception
                    centroids
                    error('Problem\n');
                end
                
                %                 [groups, ~, ~] = fast_kmeans(points, centroids, 1);
                
                [group_ids, ~, j] = unique(groups);
                group_ids = 1:length(group_ids);
                groups = group_ids(j)';
                %                 fac = max([1, ceil(64 / max(groups(rhyt_idx))) - 1]);
                %                 fac*max(groups(rhyt_idx))
                %                 figure(1); scatter(states(rhyt_idx, 1), states(rhyt_idx, 2), [], col(round(groups(rhyt_idx) * fac), :), 'filled');
            end
            warning('on');
            %             fprintf('    %i groups\n', length(unique(groups)));
        end
        
        function  [m, n, r, weight] = viterbi_filtering(obj, m, n, r, weight)
            % compute tempo matrix: rows are particles at iFrame-1,
            % cols are particles at iFrame
            tempo_current = bsxfun(@minus, m(:, iFrame), m(:, iFrame-1)')';
            rhythm_constant = bsxfun(@eq, r(:, iFrame), r(:, iFrame-1)')';
            % add Meff to negative tempi
            for iR=1:obj.R
                rhythm_iR = bsxfun(@eq, r(:, iFrame), (ones(obj.nParticles, 1)*iR)')';
                idx = rhythm_constant & rhythm_iR & (tempo_current < 0);
                tempo_current(idx) = tempo_current(idx) + obj.Meff(obj.rhythm2meter_state(iR));
            end
            tempo_prev = n(:, iFrame-1);
            tempoChange = bsxfun(@minus, tempo_current, tempo_prev) / obj.M;
            logTransProbCont = log(zeros(size(tempoChange)));
            transition_ok = tempoChange < (10 * obj.sigma_N);
            logTransProbCont(rhythm_constant & transition_ok) = ...
                log(normpdf(tempoChange(rhythm_constant & transition_ok) , 0, obj.sigma_N));
            % find best precursor particle
            [weight, i_part] = max(bsxfun(@plus, logTransProbCont, weight(:)));
            m(:, 1:iFrame-1) = m(i_part, 1:iFrame-1);
            n(:, 1:iFrame-1) = n(i_part, 1:iFrame-1);
        end
        
        function [weight, groups, newIdx] = resample(obj, weight, groups)
            if obj.resampling_scheme == 0
                newIdx = obj.resampleSystematic(exp(weight));
                weight = log(ones(obj.nParticles, 1) / obj.nParticles);  
            elseif obj.resampling_scheme == 1 % APF
                % warping:
                w = exp(weight);
                f = str2func(obj.warp_fun);
                w_warped = f(w);
                newIdx = obj.resampleSystematic(w_warped);
                w_fac = w ./ w_warped;
                weight = log( w_fac(newIdx) / sum(w_fac(newIdx)) );  
            elseif obj.resampling_scheme == 2 % K-MEANS
                % k-means clustering
                [newIdx, weight, groups] = obj.resample_in_groups2(groups, weight, obj.n_max_clusters);
                weight = weight';
            elseif obj.resampling_scheme == 3 % APF + K-MEANS
                % apf and k-means
                [newIdx, weight, groups] = obj.resample_in_groups2(groups, weight, obj.n_max_clusters, str2func(obj.warp_fun));
                weight = weight';
            else
                fprintf('WARNING: Unknown resampling scheme!\n');
            end
        end
        
    end
    
    methods (Static)
        %         outIndex = systematicR(inIndex,wn);
        outIndex = resampleSystematic( w, n_samples );
        
        outIndex = resampleSystematic2( w, n_samples );
        
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
        
        [outIndex, outWeights, groups] = resample_in_groups(groups, weights, n_max_clusters, warp_fun);
        
        [outIndex, outWeights, groups] = resample_in_groups2(groups, weights, n_max_clusters, warp_fun);
        
    end
    
    
    
end
