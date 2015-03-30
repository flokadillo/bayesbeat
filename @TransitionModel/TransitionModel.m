classdef TransitionModel
    % Transition Model Class
    properties (SetAccess=public)
        tempo_transition_probs % [N*R x N]
    end
    
    properties (SetAccess=private)
        A               % sparse transition matrix [nStates x nStates]
        M               % number of max bar positions
        N               % number of tempo states
        R               % number of rhythmic pattern states
        pn              % probability of a switch in tempo
        pr              % probability of a switch in rhythmic pattern
        rhythm2meter_state    % assigns each rhythmic pattern to a meter state
        minN            % min tempo (n_min) for each rhythmic pattern
        maxN            % max tempo (n_max) for each rhythmic pattern
        state_type      % 'discrete' or 'continuous'
        use_silence_state % [true/false] use silence stateto pause tracker
        alpha           % squeezing factor for the tempo change distribution
        M_per_pattern   % [1 x R] effective number of bar positions per rhythm pattern
        num_beats_per_pattern % [1 x R] number of beats per pattern
        frames_per_beat % cell arrays with tempi relative to the framerate
                        % frames_per_beat{1} is a row vector with tempo
                        % values in [audio frames per beat] for pattern 1
        num_position_states_per_beat % number of position states per beat
        num_position_states_per_pattern % cell array of length R
        p2s                 % prior probability to go into silence state
        pfs                 % prior probability to exit silence state
        mapping_state_tempo     % [n_states, 1] contains for each state the
                                %  corresponding tempo
        mapping_state_position  % [n_states, 1] contains for each state the
                                %  corresponding position
        mapping_state_rhythm    % [n_states, 1] contains for each state the
                                % corresponding rhythm
        mapping_tempo_state_id      % [n_states, 1] contains the tempo state number
        mapping_position_state_id   % [n_states, 1] contains the position state number
        mapping_substates_state % [M, N, R]
        proximity_matrix        % [n_states, 6] states id of neighboring states
                                %   in the order left, left down, right down,
                                %   right, right up, left up
        tempo_state
        num_states
    end
    
    
    
    methods
        function obj = TransitionModel(M_per_pattern, N, R, pn, pr, alpha, ...
                position_states_per_beat, frames_per_beat, use_silence_state, ...
                p2s, pfs, tm_type)
            % save parameters
            obj.M = max(M_per_pattern);
            obj.M_per_pattern = M_per_pattern;
            obj.N = N;
            obj.R = R;
            obj.pn = pn;
            obj.pr = pr;
            obj.alpha = alpha;
            obj.num_position_states_per_beat = position_states_per_beat;
            obj.num_beats_per_pattern = round(M_per_pattern ./ ...
                position_states_per_beat);
            obj.frames_per_beat = frames_per_beat;
            
            % save silence state variables
            if exist('use_silence_state', 'var')
                obj.use_silence_state = use_silence_state;
                obj.p2s = p2s;
                obj.pfs = pfs;
            else
                obj.use_silence_state = 0;
            end
            if ~obj.use_silence_state
                obj.p2s = 0;
                obj.pfs = 0;
            end
            fprintf('* Set up transition model .');
            % compute tempo range in frame domain
            for ri = 1:obj.R
                tempo_states = obj.num_position_states_per_beat ./ ...
                    obj.frames_per_beat{ri};
                obj.minN(ri) = min(tempo_states);
                obj.maxN(ri) = max(tempo_states);
            end
            if ~exist('tm_type', 'var')
                tm_type = 'whiteley';
            end
            if strcmp(tm_type, 'whiteley')
                obj = obj.make_whiteleys_tm(1);
            elseif strcmp(tm_type, '2015')
                obj = obj.make_2015_tm();
            end
            fprintf('done\n');
        end
        
        
        function obj = make_whiteleys_tm(obj, do_output)
            % This function creates a transition matrix as proposed by
            % N.Whiteley et al.. "Bayesian Modelling of Temporal Structure
            % in Musical Audio." ISMIR. 2006.
            % set up pattern transition matrix
            if obj.use_silence_state
                silence_state_id = obj.M * obj.N *obj.R + 1;
            end
            obj.num_states = obj.M * obj.N * obj.R;
            % alloc memory for
            obj.mapping_state_tempo = ones(obj.num_states, 1) * (-1);
            obj.mapping_state_position = ones(obj.num_states, 1) * (-1);
            obj.mapping_state_rhythm = ones(obj.num_states, 1) * (-1);
            if (length(obj.pr(:)) == 1) && (obj.R > 1)
                % expand pr to a matrix [R x R]
                % transitions to other patterns
                pr_mat = ones(obj.R, obj.R) * (obj.pr / (obj.R-1));
                % pattern self transitions
                pr_mat(logical(eye(obj.R))) = (1-obj.pr);
            elseif (size(obj.pr, 1) == obj.R) && (size(obj.pr, 2) == obj.R)
                % ok, do nothing
            else
                error('p_r has wrong dimensions!\n');
            end
            % set up tempo transition matrix
            if size(obj.pn, 1) == obj.R * obj.N
                n_r_trans = obj.pn;
            elseif size(obj.pn, 1) == 2
                n_r_trans = zeros(obj.R * obj.N, obj.N);
                for ri = 1:obj.R
                    n_const = diag(ones(obj.N, 1) * (1-sum(obj.pn)), 0);
                    n_up = diag(ones(obj.N, 1) * obj.pn(1), 1);
                    n_down = diag(ones(obj.N, 1) * obj.pn(2), -1);
                    n_r_trans((ri-1) * obj.N + 1:ri * obj.N, :) = ...
                        n_const + n_up(1:obj.N, 1:obj.N) + n_down(1:obj.N, 1:obj.N);
                end
            elseif size(obj.pn, 1) == 1 % prob of tempo increase and decrease are the same (pn/2)
                n_r_trans = zeros(obj.R * obj.N, obj.N);
                for ri = 1:obj.R
                    n_const = diag(ones(obj.N, 1) * (1-2*obj.pn), 0);
                    n_up = diag(ones(obj.N, 1) * obj.pn, 1);
                    n_down = diag(ones(obj.N, 1) * obj.pn, -1);
                    n_r_trans((ri-1) * obj.N + 1:ri * obj.N, :) = ...
                        n_const + n_up(1:obj.N, 1:obj.N) + n_down(1:obj.N, 1:obj.N);
                end
            else
                error('p_n has wrong dimensions!\n');
            end
            p = 1;
            % memory allocation:
            ri = zeros(obj.num_states*3,1); cj = zeros(obj.num_states*3,1); val = zeros(obj.num_states*3,1);
            for rhi = 1:obj.R
                if do_output, fprintf('.'); end;
                mi=1:obj.M_per_pattern(rhi);
                for ni = obj.minN(rhi)+1:obj.maxN(rhi)-1
                    % decode m, n, r into state index i
                    i = sub2ind([obj.M, obj.N, obj.R], mi, repmat(ni, 1, ...
                        obj.M_per_pattern(rhi)), ...
                        repmat(rhi, 1, obj.M_per_pattern(rhi)));
                    % save state mappings
                    obj.mapping_state_tempo(i) = ni;
                    obj.mapping_state_position(i) = mi;
                    obj.mapping_state_rhythm(i) = rhi;
                    % position of state j
                    mj = mod(mi + ni - 1, obj.M_per_pattern(rhi)) + 1; % new position
                    % --------------------------------------------------------------
                    bar_crossing = mj < mi;
                    n_bc = sum(bar_crossing);
                    nn_bc = length(bar_crossing) - n_bc;
                    % possible transitions: 3 x T x R
                    ind_rn = (rhi - 1) * obj.N + ni;
                    for n_ind = 1:3
                        if n_ind == 1 % tempo decrease
                            nj = ni - 1;
                            j_n = mj(bar_crossing) + (nj - 1) * obj.M;
                        elseif n_ind == 2 % tempo constant
                            nj = ni;
                            j_n = mj(bar_crossing) + (nj - 1) * obj.M;
                        else  % tempo increase
                            nj = ni+1;
                            j_n = mj(bar_crossing) + (nj - 1) * obj.M;
                        end
                        prob = n_r_trans(ind_rn, nj);
                        for rhj=1:obj.R
                            prob2 = pr_mat(rhi, rhj);
                            j = (rhj - 1) * obj.N * obj.M + j_n;
                            ri(p:p+n_bc-1) = i(bar_crossing);
                            cj(p:p+n_bc-1) = j;
                            val(p:p+n_bc-1) = prob * prob2 * (1-obj.p2s);
                            p = p + n_bc;
                        end
                    end
                    if obj.use_silence_state
                        % transition to silence state possible at bar
                        % transition
                        ri(p:p+n_bc-1) = i(bar_crossing);
                        cj(p:p+n_bc-1) = silence_state_id;
                        val(p:p+n_bc-1) = obj.p2s;
                        p = p + n_bc;
                    end
                    % --------------------------------------------------------------
                    % inside the bar
                    j_mr = (rhi - 1) * obj.N * obj.M + mj(~bar_crossing);
                    % possible transitions: 3
                    for n_ind = 1:3 % decrease, constant, increase
                        if n_ind == 1 % tempo decrease
                            nj = ni - 1;
                        elseif n_ind == 2 % tempo constant
                            nj = ni;
                        else  % tempo increase
                            nj = ni+1;
                        end
                        prob = n_r_trans((rhi-1)*obj.N + ni, nj);
                        j = (nj - 1) * obj.M + j_mr;
                        ri(p:p+nn_bc-1) = i(~bar_crossing);
                        cj(p:p+nn_bc-1) = j;
                        val(p:p+nn_bc-1) = prob;
                        p = p + nn_bc;
                    end
                end
            end
            % --------------------------------------------------------------
            % set probabilities for states with min and max tempo (borders of state space)
            j_r = (0:obj.R-1) * obj.N * obj.M;
            mi = 1:obj.M;
            for rhi=1:obj.R
                % -----------------------------------------------
                % ni = minimal tempo:
                % -----------------------------------------------
                ni = obj.minN(rhi);
                % only 2 possible transitions
                % 1) tempo constant
                nj = ni;
                mj = mod(mi + ni - 1, obj.M_per_pattern(rhi)) + 1; % new position
                i = j_r(rhi) + (ni - 1) * obj.M + mi;
                j = j_r(rhi) + (nj - 1) * obj.M + mj;
                ri(p:p+obj.M-1) = i;  cj(p:p+obj.M-1) = j;
                val(p:p+obj.M-1) = n_r_trans((rhi-1)*obj.N + ni, nj);
                p = p + obj.M;
                % save state mappings
                obj.mapping_state_tempo(i) = ni;
                obj.mapping_state_position(i) = mi;
                obj.mapping_state_rhythm(i) = rhi;
                % 2) tempo increase
                j = j + obj.M;
                ri(p:p+obj.M-1) = i;  cj(p:p+obj.M-1) = j;
                val(p:p+obj.M-1) = 1 - val(p-obj.M:p-1);   p = p + obj.M;
                % -----------------------------------------------
                % ni = maximal tempo:
                % -----------------------------------------------
                ni = min([obj.maxN(rhi), obj.N]);
                % only 2 possible transitions
                % 1) tempo constant
                nj = ni;
                i = j_r(rhi) + (ni-1)*obj.M + mi;
                ri(p:p+obj.M-1) = i;  cj(p:p+obj.M-1) = j;
                val(p:p+obj.M-1) = n_r_trans((rhi-1)*obj.N + ni, nj);
                p = p + obj.M;
                % save state mappings
                obj.mapping_state_tempo(i) = ni;
                obj.mapping_state_position(i) = mi;
                obj.mapping_state_rhythm(i) = rhi;
                % 2) tempo decrease
                j = j - obj.M;
                ri(p:p+obj.M-1) = i;  cj(p:p+obj.M-1) = j;
                val(p:p+obj.M-1) = 1 - val(p-obj.M:p-1);   p = p + obj.M;
            end
            
            if obj.use_silence_state
                p0 = p;
                % self transition
                cj(p) = silence_state_id;
                val(p) = 1 - obj.pfs;
                p = p + 1;
                % transition from silence state to m=1, n(:), r(:)
                n = [];
                r = [];
                for i_r=1:obj.R
                    n = [n, obj.minN(i_r):obj.maxN(i_r)];
                    r = [r, ones(1, length(obj.minN(i_r):obj.maxN(i_r))) * i_r];
                end
                cj(p:p+length(n(:))-1) = ones(length(n(:)), 1) + ...
                    (n(:)-1)*obj.M + (r(:)-1)*obj.M*obj.N;
                val(p:p+length(n(:))-1) = obj.pfs/(length(n(:)));
                p = p + length(n(:));
                ri(p0:p-1) = silence_state_id;
                obj.A = sparse(ri(1:p-1), cj(1:p-1), val(1:p-1), ...
                    obj.num_states+1, obj.num_states+1);
            else
                obj.A = sparse(ri(1:p-1), cj(1:p-1), val(1:p-1), ...
                    obj.num_states, obj.num_states);
            end
            obj.mapping_position_state_id = obj.mapping_state_position;
            obj.mapping_tempo_state_id = obj.mapping_state_tempo;
        end
        
        function obj = make_2015_tm(obj)
            % This function creates a transition function with the following
            % properties: each tempostate has a different number of position
            % states
            % only allow transition with probability above threshold
            threshold = eps;
            
            % total number of states
            obj.num_states = cellfun(@(x) sum(x), obj.frames_per_beat) * ...
                obj.num_beats_per_pattern(:);
            num_tempo_states = cellfun(@(x) length(x), obj.frames_per_beat);
            % attach silence state to the end of the states
            if obj.use_silence_state
                silence_state_id = obj.num_states + 1;
                obj.num_states = obj.num_states + 1;
            end
            max_n_pos_states = max(cellfun(@(x) sum(x), obj.frames_per_beat) .* ...
                obj.num_beats_per_pattern);
            max_pos_states_of_pattern = max(cellfun(@(x) max(x), obj.frames_per_beat) .* ...
                obj.num_beats_per_pattern);
            max_n_tempo_states = max(cellfun(@(x) length(x), obj.frames_per_beat));
            % compute mapping between linear state index and bar position,
            % tempo and rhythmic pattern sub-states
            % state index counter
            obj.mapping_state_position = ones(obj.num_states, 1) * (-1);
            obj.mapping_state_tempo = ones(obj.num_states, 1) * (-1);
            obj.mapping_state_rhythm = ones(obj.num_states, 1, 'int32') * (-1);
            obj.mapping_position_state_id = ones(obj.num_states, 1, 'int32') * (-1);
            obj.mapping_tempo_state_id = ones(obj.num_states, 1, 'int32') * (-1);
            obj.mapping_substates_state = ones(max_n_pos_states, ...
                max_n_tempo_states, obj.R, 'int32') * (-1);
            obj.num_position_states_per_pattern = cell(obj.R, 1);
            obj.proximity_matrix = ones(obj.num_states, 6, 'int32') * (-1);
            si = 1;
            for ri = 1:obj.R
                obj.num_position_states_per_pattern{ri} = ...
                    zeros(num_tempo_states(ri), 1);
                n_pos_states = obj.frames_per_beat{ri} * ...
                    obj.num_beats_per_pattern(ri);
                for tempo_state_i = 1:num_tempo_states(ri)
                    idx = si:(si+n_pos_states(tempo_state_i)-1);
                    obj.mapping_state_rhythm(idx) = ri;
                    obj.mapping_state_tempo(idx) = obj.num_position_states_per_beat ./ ...
                        obj.frames_per_beat{ri}(tempo_state_i);
                    obj.mapping_tempo_state_id(idx) = tempo_state_i;
                    obj.mapping_state_position(idx) = ...
                        (0:(n_pos_states(tempo_state_i) - 1)) .* ...
                        obj.M_per_pattern(ri) ./ n_pos_states(tempo_state_i) + 1;
                    obj.mapping_position_state_id(idx) = 1:n_pos_states(tempo_state_i);
                    obj.num_position_states_per_pattern{ri}(tempo_state_i) = ...
                        n_pos_states(tempo_state_i);
                    for i = 1:n_pos_states(tempo_state_i)
                        obj.mapping_substates_state(i, tempo_state_i, ri) = idx(i);
                    end
                    % set up proximity matrix
                    % states to the left
                    obj.proximity_matrix(idx, 1) = [idx(end), idx(1:end-1)];
                    % states to the right
                    obj.proximity_matrix(idx, 4) = [idx(2:end), idx(1)];
                    % states to down
                    if tempo_state_i > 1
                        state_id = (0:n_pos_states(tempo_state_i)-1) * ...
                            n_pos_states(tempo_state_i - 1) /  ...
                            n_pos_states(tempo_state_i) + 1;
                        s_start = idx(1) - n_pos_states(tempo_state_i - 1);
                        % left down
                        temp = floor(state_id);
                        % modulo operation
                        temp(temp == 0) = n_pos_states(tempo_state_i - 1);
                        obj.proximity_matrix(idx, 2) = temp + s_start - 1;
                        % right down
                        temp = ceil(state_id);
                        % modulo operation
                        temp(temp > n_pos_states(tempo_state_i - 1)) = 1;
                        obj.proximity_matrix(idx, 3) = temp + s_start - 1;
                        % if left down and right down are equal set to -1
                        obj.proximity_matrix(find(rem(state_id, 1) == 0) + ...
                            + idx(1) - 1, 3) = -1;
                    end
                    % states to up
                    if tempo_state_i < num_tempo_states(ri)
                        % for each position state of the slow tempo find the
                        % corresponding state of the faster tempo
                        state_id = (0:n_pos_states(tempo_state_i)-1) * ...
                            n_pos_states(tempo_state_i + 1) /  ...
                            n_pos_states(tempo_state_i) + 1;
                        % left up
                        temp = floor(state_id);
                        % modulo operation
                        temp(temp == 0) = n_pos_states(tempo_state_i + 1);
                        obj.proximity_matrix(idx, 6) = temp + idx(end);
                        % right up
                        temp = ceil(state_id);
                        % modulo operation
                        temp(temp > n_pos_states(tempo_state_i + 1)) = 1;
                        obj.proximity_matrix(idx, 5) = temp + idx(end);
                        % if left up and right up are equal set to -1
                        obj.proximity_matrix(find(rem(state_id, 1) == 0) + ...
                            idx(1) - 1, 6) = -1;
                    end
                    si = si + length(idx);
                end
            end
            
            if obj.use_silence_state
                obj.mapping_state_tempo(silence_state_id) = 0;
                obj.mapping_state_position(silence_state_id) = 0;
                obj.mapping_state_rhythm(silence_state_id) = obj.R + 1;
                obj.mapping_position_state_id(silence_state_id) = 0;
                obj.mapping_tempo_state_id(silence_state_id) = 0;
            end
            
            % save the highest number of tempo states of all patterns
            obj.N = max(num_tempo_states);
            % tempo changes can only occur at the beginning of a beat
            % compute transition matrix for the tempo changes
            trans_prob = cell(obj.R);
            % iterate over all tempo states
            for ri = 1:obj.R
                trans_prob{ri} = zeros(num_tempo_states(ri), num_tempo_states(ri));
                for tempo_state_i = 1:length(obj.frames_per_beat{ri})
                    for tempo_state_j = 1:length(obj.frames_per_beat{ri})
                        % compute the ratio of the number of beat states between the
                        % two tempi
                        ratio = obj.frames_per_beat{ri}(tempo_state_j) /...
                            obj.frames_per_beat{ri}(tempo_state_i);
                        % compute the probability for the tempo change
                        prob = exp(-obj.alpha * abs(ratio - 1));
                        % keep only transition probabilities > threshold
                        if prob > threshold
                            % save the probability
                            trans_prob{ri}(tempo_state_i, tempo_state_j) = prob;
                        end
                    end
                    % normalise
                    trans_prob{ri}(tempo_state_i, :) = ...
                        trans_prob{ri}(tempo_state_i, :) ./ ...
                        sum(trans_prob{ri}(tempo_state_i, :));
                end
            end
            % Apart from the very beginning of a beat, the tempo stays the same,
            % thus the number of transitions is equal to the total number of states
            % plus the number of tempo transitions minus the number of tempo states
            % since these transitions are already included in the tempo transitions
            % Then everything multiplicated with the number of beats with
            % are modelled in the patterns
            % TODO: Note changes between patterns are not implemented yet!
            num_tempo_transitions = (num_tempo_states .* num_tempo_states) * ...
                obj.num_beats_per_pattern(:);
            if obj.use_silence_state
                num_transitions = obj.num_states + num_tempo_transitions - ...
                    (num_tempo_states * obj.num_beats_per_pattern(:)) + ...
                    2 * sum(num_tempo_states) + 1;
            else
                num_transitions = obj.num_states + num_tempo_transitions - ...
                    (num_tempo_states * obj.num_beats_per_pattern(:));
            end
            % initialise vectors to store the transitions in sparse format
            % rows (states at previous time)
            row_i = zeros(num_transitions, 1);
            % cols (states at current time)
            col_j = zeros(num_transitions, 1);
            % transition probabilites
            val = zeros(num_transitions, 1);
            
            num_states_per_pattern = cellfun(@(x) sum(x), obj.frames_per_beat) .* ...
                obj.num_beats_per_pattern;
            % get linear index of all beat positions
            positions_at_beat = cell(obj.R, max(obj.num_beats_per_pattern));
            si = 0;
            for ri = 1:obj.R
                positions_at_beat{ri, 1} = si + cumsum([1, ...
                    obj.frames_per_beat{ri}(1:end-1) * ...
                    obj.num_beats_per_pattern(ri)]);
                
                for bi = 2:obj.num_beats_per_pattern(ri)
                    positions_at_beat{ri, bi} = ...
                        positions_at_beat{ri, bi-1} + obj.frames_per_beat{ri};
                end
                si = si + num_states_per_pattern(ri);
            end
            % get linear index of preceeding state of all beat positions
            positions_before_beat = cell(obj.R, max(obj.num_beats_per_pattern));
            for ri = 1:obj.R
                for bi = 2:obj.num_beats_per_pattern(ri)
                    positions_before_beat{ri, bi} = ...
                        positions_at_beat{ri, bi} - 1;
                end
                positions_before_beat{ri, 1} =  ...
                    positions_at_beat{ri, obj.num_beats_per_pattern(ri)} + ...
                    obj.frames_per_beat{ri} - 1;
            end
            % transition counter
            p = 1;
            for ri = 1:obj.R
                for bi = 1:obj.num_beats_per_pattern(ri)
                    for ni = 1:num_tempo_states(ri)
                        for nj = 1:num_tempo_states(ri)
                            if trans_prob{ri}(ni, nj) ~= 0
                                % create transition
                                % position before beat
                                row_i(p) = positions_before_beat{ri, bi}(ni);
                                % position at beat
                                col_j(p) = positions_at_beat{ri, bi}(nj);
                                % store probability
                                val(p) = trans_prob{ri}(ni, nj);
                                p = p + 1;
                            end
                        end % over tempo at current time
                        % transitions of the remainder of the beat: tempo
                        % transitions are not allowed
                        idx = p:p+obj.frames_per_beat{ri}(ni) - 2;
                        row_i(idx) = positions_at_beat{ri, bi}(ni) + ...
                            (0:obj.frames_per_beat{ri}(ni)-2);
                        col_j(idx) = positions_at_beat{ri, bi}(ni) + ...
                            (1:obj.frames_per_beat{ri}(ni)-1);
                        val(idx) = 1;
                        p = p + length(idx);
                    end % over tempo at previous time
                end % over beats
                if obj.use_silence_state
                    % transition to silence state possible at bar
                    % transition
                    idx = p:(p + num_tempo_states(ri) - 1);
                    row_i(idx) = positions_before_beat{ri, 1}(:);
                    col_j(idx) = silence_state_id;
                    val(idx) = obj.p2s;
                    p = p + num_tempo_states(ri);
                end
            end % over R
            
            % add transitions from silence state
            if obj.use_silence_state
                % one self transition and one transition to each R
                idx = p:(p + sum(num_tempo_states));
                % start at silence
                row_i(idx) = silence_state_id;
                % self transition
                col_j(p) = silence_state_id;
                val(p) = 1 - obj.pfs;
                p = p + 1;
                % transition from silence state to m=1, n(:), r(:)
                prob_from_silence = obj.pfs / sum(num_tempo_states);
                for i_r=1:obj.R
                    idx = p:(p + num_tempo_states(i_r) - 1);
                    % go to first position of each tempo
                    col_j(idx) = positions_at_beat{i_r, 1};
                    val(idx) = prob_from_silence;
                    p = p + num_tempo_states(i_r);
                end
                
            end
            idx = (row_i > 0);
            obj.A = sparse(row_i(idx), col_j(idx), val(idx), ...
                obj.num_states, obj.num_states);
        end
        
        function error = transition_model_is_corrupt(obj, dooutput)
            if nargin == 1, dooutput = 0; end
            error = 1;
            if dooutput, fprintf('*   Checking the Transition Matrix ... '); end
            % sum over columns j
            sum_over_j = full(sum(obj.A, 2));
            % find corrupt states: sum_over_j should be either 0 (if state is
            % never visited) or 1
            zero_probs_j = abs(sum_over_j) < 1e-4;  % p ≃ 0
            one_probs_j = abs(sum_over_j-1) < 1e-4; % p ≃ 1
            corrupt_states_i = find(~zero_probs_j & ~one_probs_j, 1);
            
            if dooutput
                fprintf('    Number of non-zero states: %i (%.1f %%)\n', ...
                    sum(sum_over_j==1), sum(sum_over_j==1)*100/size(obj.A,1));
                memory = whos('obj.A');
                fprintf('    Memory: %.3f MB\n',memory.bytes/1e6);
            end
            
            if isempty(corrupt_states_i)
                error = 0;
                if dooutput, fprintf('done\n'); end
            else
                fprintf('    Number of corrupt states: %i  ',length(corrupt_states_i));
                if ~isempty(corrupt_states_i)
                    fprintf('    (%i - %i)\n',corrupt_states_i(1), corrupt_states_i(end));
                    m = obj.mapping_position_state_id(corrupt_states_i(1));
                    n = obj.mapping_tempo_state_id(corrupt_states_i(1));
                    r = obj.mapping_state_rhythm(corrupt_states_i(1));
                    fprintf('    State %i (%i - %i - %i) has transition TO: \n', ...
                        corrupt_states_i(1), m, n, r);
                    trans_states = find(obj.A(corrupt_states_i(1), :));
                    sumProb = 0;
                    for i=1:length(trans_states)
                        [m, n, r] =  ind2sub([obj.M obj.M obj.R], trans_states(i));
                        fprintf('      %i (%i - %i - %i) with p=%.10f\n', ...
                            trans_states(i), m, n, r, ...
                            full(obj.A(corrupt_states_i(1), trans_states(i))));
                        sumProb = sumProb + full(obj.A(corrupt_states_i(1), ...
                            trans_states(i)));
                    end
                    fprintf('      sum: p=%.10f\n', sumProb);
                end
            end
        end % transition_model_is_corrupt
    end
end
