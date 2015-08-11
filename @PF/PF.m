classdef PF < handle
    % Hidden Markov Model Class
    properties (SetAccess=private)
        M                           % number of positions in a 4/4 bar
        Meff                        % number of positions per rhythm [R x 1]
        N                           % number of tempo states
        R                           % number of rhythmic pattern states
        T                           % number of meters
        pr                          % probability of a switch in rhythmic pattern
        rhythm2meter                % assigns each rhythmic pattern to a
        %                           meter [R x 1]
        rhythm2meter_state          % assigns each rhythmic pattern to a
        % meter state (1, 2, ...)  - var not needed anymore but keep due to
        % compatibility
        meter_state2meter           % specifies meter for each meter state
        % (9/8, 8/8, 4/4) [2 x nMeters] - var not needed anymore but keep due to
        % compatibility
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
        n_depends_on_r              % Flag to say if tempo depends on the style/pattern/rhythm
    end
    
    methods
        function obj = PF(Params, rhythm2meter, rhythm_names)
            obj.M = Params.M;
            obj.N = Params.N;
            obj.R = Params.R;
            obj.T = size(Params.meters, 2);
            obj.nParticles = Params.nParticles;
            %             obj.sigma_N = Params.sigmaN; % moved this to
            %             make_transition_model
            obj.barGrid = max(Params.whole_note_div * (Params.meters(1, :) ...
                ./ Params.meters(2, :)));
            obj.frame_length = Params.frame_length;
            obj.dist_type = Params.observationModelType;
            obj.rhythm2meter = rhythm2meter;
            bar_durations = rhythm2meter(:, 1) ./ rhythm2meter(:, 2);
            obj.Meff = round((bar_durations) ...
                * (Params.M ./ (max(bar_durations)))); obj.Meff = obj.Meff(:);
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
            obj.n_depends_on_r = Params.n_depends_on_r;
            tempVar = ver('matlab');
            if str2double(tempVar.Release(3:6)) < 2011
                % Matlab 2010
                disp('MATLAB 2010');
                RandStream.setDefaultStream(RandStream('mt19937ar','seed', ...
                    sum(100*clock)));
            else
                rng('shuffle');
            end
        end
        
        function obj = make_initial_distribution(obj, tempo_per_cluster)
            
            % TODO: Implement prior initial distribution for tempo
            obj.initial_m = zeros(obj.nParticles, 1);
            obj.initial_n = zeros(obj.nParticles, 1);
            obj.initial_r = zeros(obj.nParticles, 1);
            
            % use pseudo random monte carlo
            n_grid = min(obj.minN):max(obj.maxN);
            n_m_cells = floor(obj.nParticles / length(n_grid));
            m_grid_size = sum(obj.Meff) / n_m_cells;
            r_m = rand(obj.nParticles, 1) - 0.5; % between -0.5 and +0.5
            r_n = rand(obj.nParticles, 1) - 0.5;
            nParts = zeros(obj.R, 1);
            c=1;
            for iR = 1:obj.R
                % create positions between 1 and obj.Meff(obj.rhythm2meter_state(iR))
                m_grid = (1+m_grid_size/2):m_grid_size:...
                    (obj.Meff(iR)-m_grid_size/2);
                n_grid_iR = obj.minN(iR):obj.maxN(iR);
                nParts(iR) = length(m_grid) * length(n_grid_iR);
                temp = repmat(m_grid, length(n_grid_iR), 1);
                obj.initial_m(c:c+nParts(iR)-1) = temp(:)+ ...
                    r_m(c:c+nParts(iR)-1) * m_grid_size;
                obj.initial_n(c:c+nParts(iR)-1) = repmat(n_grid_iR, 1, ...
                    length(m_grid))' + r_n(c:c+nParts(iR)-1);
                obj.initial_r(c:c+nParts(iR)-1) = iR;
                c = c + nParts(iR);
            end
            if sum(nParts) < obj.nParticles
                % add remaining particles randomly
                % random pattern assignment
                obj.initial_r(c:end) = round(rand(obj.nParticles+1-c, 1)) ...
                    * (obj.R-1) + 1;
                n_between_0_and_1 = (r_n(c:end) + 0.5)';
                m_between_0_and_1 = (r_m(c:end) + 0.5);
                % map n_between_0_and_1 to allowed tempo range
                obj.initial_n(c:end) = n_between_0_and_1 .* obj.maxN(...
                    obj.initial_r(c:end)) + (1 - n_between_0_and_1) .* ...
                    obj.minN(obj.initial_r(c:end));
                % map m_between_0_and_1 to allowed position range
                obj.initial_m(c:end) = m_between_0_and_1 .* ...
                    (obj.Meff(obj.initial_r(c:end))-1) + 1;
            end
        end
        
        function obj = make_transition_model(obj, minTempo, maxTempo, ...
                alpha, sigmaN, pr)
            
            obj.sigma_N = sigmaN;
            % convert from BPM into barpositions / audio frame
            meter_num = obj.rhythm2meter(:, 1);
            % save pattern change probability and save as matrix [RxR]
            if (length(pr(:)) == 1) && (obj.R > 1)
                % expand pr to a matrix [R x R]
                % transitions to other patterns
                pr_mat = ones(obj.R, obj.R) * (pr / (obj.R-1));
                % pattern self transitions
                pr_mat(logical(eye(obj.R))) = (1-pr);
                obj.pr = pr_mat;
            elseif (size(pr, 1) == obj.R) && (size(pr, 2) == obj.R)
                % ok, do nothing
                obj.pr = pr;
            else
                error('p_r has wrong dimensions!\n');
            end
            if strcmp(obj.pattern_size, 'bar')
                obj.minN = floor(obj.Meff .* obj.frame_length .* minTempo ./ (meter_num * 60));
                obj.maxN = ceil(obj.Meff .* obj.frame_length .* maxTempo ./ (meter_num * 60));
            else
                obj.minN = floor(obj.M * obj.frame_length * minTempo ./ 60);
                obj.maxN = ceil(obj.M * obj.frame_length * maxTempo ./ 60);
            end
            if max(obj.maxN) > obj.N
                fprintf('    N should be %i instead of %i -> corrected\n', ...
                    max(obj.maxN), obj.N);
                obj.N = ceil(max(obj.maxN));
            end
            if ~obj.n_depends_on_r % no dependency between n and r
                obj.minN = ones(1, obj.R) * min(obj.minN);
                obj.maxN = ones(1, obj.R) * max(obj.maxN);
                obj.N = max(obj.maxN);
                fprintf('    Tempo limited to %i - %i bpm\n', ...
                    round(min(obj.minN)*60*4/(obj.M * obj.frame_length)), ...
                    round(max(obj.maxN)*60*4/(obj.M * obj.frame_length)));
            end
            % Transition function to propagate particles
            obj.sample_trans_fun = @(x) obj.propagate_particles_pf(obj, x);
        end
        
        function obj = make_observation_model(obj, train_data)
            
            % Create observation model
            obj.obs_model = ObservationModel(obj.dist_type, obj.rhythm2meter, ...
                obj.M, obj.N, obj.R, obj.barGrid, obj.Meff, ...
                train_data.feat_type, obj.use_silence_state);
            
            % Train model
            obj.obs_model = obj.obs_model.train_model(train_data);
            
            obj.train_dataset = train_data.dataset;
            
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
        
        function results = do_inference(obj, y, fname, inference_method, do_output)
            if isempty(strfind(inference_method, 'PF'))
                error('Inference method %s not compatible with PF model\n', inference_method);
            end
            % compute observation likelihoods
            obs_lik = obj.obs_model.compute_obs_lik(y);
            obj = obj.forward_filtering(obs_lik, fname);
            [m_path, n_path, r_path] = obj.path_via_best_weight();
            
            % strip of silence state
            if obj.use_silence_state
                idx = logical(r_path<=obj.R);
            else
                idx = true(length(r_path), 1);
            end
            % compute beat times and bar positions of beats
            meter = zeros(2, length(r_path));
            meter(:, idx) = obj.rhythm2meter(r_path(idx), :)';
            % compute beat times and bar positions of beats
            beats = obj.find_beat_times(m_path, r_path);
            if strcmp(obj.pattern_size, 'bar')
                tempo = meter(1, idx)' .* 60 .* n_path(idx)' ./ ...
                    (obj.Meff(r_path(idx)) * obj.frame_length);
            else
                tempo = 60 .* n_path(idx) / (obj.M * obj.frame_length);
            end
            results{1} = beats;
            results{2} = tempo;
            results{3} = meter;
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
        
        function obj = convert_old_model_to_new(obj)
            % check dimensions of member variables. This function might be removed
            % in future, but is important for compatibility with older models
            % (in old models Meff and
            % rhythm2meter_state are row vectors [1 x K] but should be
            % column vectors)
            obj.Meff = obj.Meff(:);
            obj.rhythm2meter_state = obj.rhythm2meter_state(:);
            % In old models, pattern change probability was not saved as
            % matrix [RxR]
            if (length(obj.pr(:)) == 1) && (obj.R > 1)
                % expand pr to a matrix [R x R]
                % transitions to other patterns
                pr_mat = ones(obj.R, obj.R) * (obj.pr / (obj.R-1));
                % pattern self transitions
                pr_mat(logical(eye(obj.R))) = (1-obj.pr);
                obj.pr = pr_mat;
            end
            if isempty(obj.rhythm2meter)
                obj.rhythm2meter = obj.meter_state2meter(:, ...
                    obj.rhythm2meter_state)';
            end
        end
        
    end
    
    methods (Access=protected)
        
        
        function lik = compute_obs_lik(obj, states_m_r, iFrame, obslik, m_per_grid)
            % states_m_r:   is a [nParts x 2] matrix, where (:, 1) are the
            %               m-values and (:, 2) are the r-values
            % obslik:       likelihood values [R, barGrid, nFrames]
            subind = floor((states_m_r(:, 1)-1) / m_per_grid) + 1;
            obslik = obslik(:, :, iFrame);
            try
                ind = sub2ind([obj.R, obj.barGrid], states_m_r(:, 2), subind(:));
                lik = obslik(ind);
            catch exception
                fprintf('dimensions R=%i, barGrid=%i, states_m=%.2f - %.2f, subind = %i - %i, m_per_grid=%.2f\n', ...
                    obj.R, obj.barGrid, min(states_m_r(:, 1)), max(states_m_r(:, 1)), ...
                    min(subind), max(subind), m_per_grid);
            end
        end
        
        function [m_path, n_path, r_path] = path_via_best_weight(obj)
            % use particle with highest weight
            % ------------------------------------------------------------
            [~, bestParticle] = max(obj.particles.weight);
            m_path = obj.particles.m(bestParticle, :);
            r_path = obj.particles.r(bestParticle, :);
            n_path = obj.particles.n(bestParticle, :)';
            m_path = m_path(:)';
            n_path = n_path(:)';
            r_path = r_path(:)';
        end
        
        
        function obj = forward_filtering(obj, obs_lik, fname)
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
            eval_lik = @(x, y) obj.compute_obs_lik(x, y, obs_lik, obj.M / ...
                obj.barGrid);
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
                m(:, iFrame) = m(:, iFrame-1) + n(:, iFrame-1);
                m(:, iFrame) = mod(m(:, iFrame) - 1, ...
                    obj.Meff(r(:, iFrame-1))) + 1;
                % Pattern transitions to be handled here
                r(:, iFrame) = r(:, iFrame-1);
                % Change the ones for which the bar changed
                crossed_barline = find(m(:, iFrame) < m(:, iFrame-1));
                for rInd = 1:length(crossed_barline)
                    r(crossed_barline(rInd), iFrame) = ...
                        randsample(obj.R, 1, true, ...
                        obj.pr(r(crossed_barline(rInd),iFrame-1),:));
                end
                % evaluate particle at iFrame-1
                obs = eval_lik([m(:, iFrame), r(:, iFrame)], iFrame);
                weight = weight(:) + log(obs(:));
                % Normalise importance weights
                [weight, ~] = obj.normalizeLogspace(weight');
                % Resampling
                % ------------------------------------------------------------
                if obj.resampling_interval == 0
                    Neff = 1/sum(exp(weight).^2);
                    do_resampling = (Neff < obj.ratio_Neff * obj.nParticles);
                else
                    do_resampling = (rem(iFrame, obj.resampling_interval) == 0);
                end
                if do_resampling && (iFrame < nFrames)
                    resampling_frames(iFrame) = iFrame;
                    if obj.resampling_scheme == 2 || obj.resampling_scheme == 3 % MPF or AMPF
                        groups = obj.divide_into_clusters([m(:, iFrame), ...
                            n(:, iFrame-1), r(:, iFrame)], ...
                            [obj.M; obj.N; obj.R], groups);
                        n_clusters(iFrame) = length(unique(groups));
                    end
                    [weight, groups, newIdx] = obj.resample(weight, groups);
                    m(:, 1:iFrame) = m(newIdx, 1:iFrame);
                    r(:, 1:iFrame) = r(newIdx, 1:iFrame);
                    n(:, 1:iFrame) = n(newIdx, 1:iFrame);
                end
                % transition from iFrame-1 to iFrame
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
            fprintf('    Average resampling interval: %.2f frames\n', ...
                mean(diff(resampling_frames(resampling_frames>0))));
            if obj.resampling_scheme > 1
                fprintf('    Average number of clusters: %.2f frames\n', mean(n_clusters(n_clusters>0)));
            end
            if obj.save_inference_data
                save(['./', fname, '_pf.mat'], 'logP_data_pf');
            end
        end
        
        function beats = find_beat_times(obj, positionPath, rhythmPath)
            %   Find beat times from sequence of bar positions of the HMM beat tracker
            % ----------------------------------------------------------------------
            %INPUT parameter:
            % positionPath             : sequence of position states
            % rhythmPath                : sequence of meter states
            %                           NOTE: so far only median of sequence is used !
            % nBarPos                  : bar length in bar positions (=M)
            % framelength              : duration of being in one state in [sec]
            %
            %OUTPUT parameter:
            %
            % beats                    : [nBeats x 2] beat times in [sec] and
            %                           beatnumber
            %
            % 29.7.2012 by Florian Krebs
            % ----------------------------------------------------------------------
            numframes = length(positionPath);
            beatpositions = cell(obj.R, 1);
            for i_r=1:obj.R
                is_compund_meter = ismember(obj.rhythm2meter(i_r, 1), ...
                    [6, 9, 12]);
                if is_compund_meter
                    beatpositions{i_r} = round(linspace(1, obj.Meff(i_r), ...
                        obj.rhythm2meter(i_r, 1) / 3 + 1));
                else % simple meter
                    beatpositions{i_r} = round(linspace(1, obj.Meff(i_r), ...
                        obj.rhythm2meter(i_r, 1) + 1));
                end
                beatpositions{i_r} = beatpositions{i_r}(1:end-1);
            end
            beatno = [];
            beats = [];
            for i = 1:numframes-1
                if rhythmPath(i) == 0
                    continue;
                end
                for beat_pos = 1:length(beatpositions{rhythmPath(i)})
                    if positionPath(i) == beatpositions{rhythmPath(i)}(beat_pos)
                        % current frame = beat frame
                        beats = [beats; [i, beat_pos]];
                        break;
                    elseif ((positionPath(i+1) > beatpositions{rhythmPath(i)}(beat_pos)) && (positionPath(i+1) < positionPath(i)))
                        % bar transition between frame i and frame i+1
                        bt = interp1([positionPath(i); obj.M+positionPath(i+1)],[i; i+1],obj.M+beatpositions{rhythmPath(i)}(beat_pos));
                        beats = [beats; [round(bt), beat_pos]];
                        break;
                    elseif ((positionPath(i) < beatpositions{rhythmPath(i)}(beat_pos)) && (beatpositions{rhythmPath(i)}(beat_pos) < positionPath(i+1)))
                        % beat position lies between frame i and frame i+1
                        bt = interp1([positionPath(i); positionPath(i+1)],[i; i+1],beatpositions{rhythmPath(i)}(beat_pos));
                        beats = [beats; [round(bt), beat_pos]];
                        break;
                    end
                end
            end
            beats(:, 1) = beats(:, 1) * obj.frame_length;
        end
        
        function [groups] = divide_into_clusters(obj, states, state_dims, groups_old)
            % states: [nParticles x nStates]
            % state_dim: [nStates x 1]
            % groups_old: [nParticles x 1] group labels of the particles
            %               after last resampling step (used for initialisation)
            warning('off');
            [group_ids, ~, IC] = unique(groups_old);
            %             fprintf('    %i groups >', length(group_ids));
            k = length(group_ids); % number of clusters
            
            % adjust the range of each state variable to make equally
            % important for the clustering
            points = zeros(obj.nParticles, length(state_dims)+1);
            points(:, 1) = (sin(states(:, 1) * 2 * pi ./ ...
                obj.Meff(states(:, 3))) + 1) * ...
                obj.state_distance_coefficients(1);
            points(:, 2) = (cos(states(:, 1) * 2 * pi ./ ...
                obj.Meff(states(:, 3))) + 1) * ...
                obj.state_distance_coefficients(1);
            points(:, 3) = states(:, 2) * ...
                obj.state_distance_coefficients(2);
            points(:, 4) =(states(:, 3)-1) * ...
                obj.state_distance_coefficients(3) + 1;
            
            % compute centroid of clusters
            % TODO: vectorise!
            centroids = zeros(k, length(state_dims)+1);
            for i_dim=1:size(points, 2)
                %                 centroids(iCluster, :) = mean(points(groups_old == group_ids(iCluster) , :));
                centroids(:, i_dim) = accumarray(IC, points(:, i_dim), [], @mean);
            end
            % do k-means clustering
            options = statset('MaxIter', 1);
            [groups, centroids, total_dist_per_cluster] = kmeans(points, k, 'replicates', 1, ...
                'start', centroids, 'emptyaction', 'drop', 'Distance', 'sqEuclidean', 'options', options);
            %             remove empty clusters
            total_dist_per_cluster = total_dist_per_cluster(~isnan(centroids(:, 1)));
            centroids = centroids(~isnan(centroids(:, 1)), :);
            [group_ids, ~, j] = unique(groups);
            group_ids = 1:length(group_ids);
            groups = group_ids(j)';
            % check if centroids are too close
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
                    if length(c1) == 1,  merging = 0;  end
                    % remove old centroid
                    centroids = centroids([1:c2(1)-1, c2(1)+1:end], :);
                    total_dist_per_cluster = total_dist_per_cluster([1:c2(1)-1, c2(1)+1:end]);
                    merged = 1;
                end
            end
            % check if cluster spread is too high
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
                split = 1;
            end
            if split || merged
                try
                    [groups, ~, ~] = kmeans(points, [], 'replicates', 1, 'start', centroids, 'emptyaction', 'drop', ...
                        'Distance', 'sqEuclidean', 'options', options);
                catch exception
                    centroids
                    error('Problem\n');
                end
                [group_ids, ~, j] = unique(groups);
                group_ids = 1:length(group_ids);
                groups = group_ids(j)';
            end
            warning('on');
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
                [newIdx, weight, groups] = obj.resample_in_groups(groups, ...
                    weight, obj.n_max_clusters);
                weight = weight';
            elseif obj.resampling_scheme == 3 % APF + K-MEANS
                % apf and k-means
                [newIdx, weight, groups] = obj.resample_in_groups(groups, ...
                    weight, obj.n_max_clusters, str2func(obj.warp_fun));
                weight = weight';
            else
                fprintf('WARNING: Unknown resampling scheme!\n');
            end
        end
    end
    
    methods (Static)
        
        outIndex = resampleSystematic( w, n_samples );
        
        outIndex = resampleSystematicInGroups( w, n_samples );
        
        function [groups] = divide_into_fixed_cells(states, state_dims, nCells)
            % divide space into fixed cells with equal number of grid
            % points for position and tempo states
            % states: [nParticles x nStates]
            % state_dim: [nStates x 1]
            groups = zeros(size(states, 1), 1);
            n_r_bins = state_dims(3);
            n_n_bins = floor(sqrt(nCells/n_r_bins));
            n_m_bins = floor(nCells / (n_n_bins * n_r_bins));
            
            m_edges = linspace(1, state_dims(1) + 1, n_m_bins + 1);
            n_edges = linspace(0, state_dims(2) + 1, n_n_bins + 1);
            for iR=1:state_dims(3)
                % get all particles that belong to pattern iR
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
        
        [y, L] = normalizeLogspace(x);
        
        r = logsumexp(X, dim);
    end
    
    
    
end
