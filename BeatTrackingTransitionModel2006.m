classdef BeatTrackingTransitionModel2006 < handle & BeatTrackingTransitionModel
    %UNTITLED4 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        pn  % probability of a tempo change
    end
    
    methods
        function obj = BeatTrackingTransitionModel2006(state_space, ...
                transition_params)
            % call superclass constructor
            obj@BeatTrackingTransitionModel(state_space, ...
                transition_params);
            obj.pn = transition_params.pn;
            obj.compute_transitions;
        end
        
    end
    
    methods (Access = protected)
        
        function [] = compute_transitions(obj)
            % This function creates a transition matrix as proposed by
            % N.Whiteley et al.. "Bayesian Modelling of Temporal Structure
            % in Musical Audio." ISMIR. 2006.
            % set up pattern transition matrix
            num_state_ids = obj.state_space.n_states;
            if obj.state_space.use_silence_state
                silence_state_id = num_state_ids + 1;
            end
            % store variables locally to avoid long variable names
            min_tempo_ss = obj.state_space.min_tempo_ss;
            max_tempo_ss = obj.state_space.max_tempo_ss;
            R = obj.state_space.n_patterns;
            N = obj.state_space.max_n_tempo_states;
            M = max(obj.state_space.max_position);
            M_from_pattern = obj.state_space.max_position;
            state_from_substate = obj.state_space.state_from_substate;
            % Set up tempo transition matrix [R*N, N], which has an NxN
            % transition matrix for each pattern            %
            if size(obj.pn, 1) == R * N
                % the tempo transition probability is assumed to differ for
                % each pattern and each absolute tempo
                n_r_trans = obj.pn;
            elseif ismember(size(obj.pn, 1), [1, 2])
                if (size(obj.pn, 1) == 1)
                    % prob of tempo increase and decrease are the same (pn)
                    obj.pn = ones(2, 1) * obj.pn;
                end
                % pn specifies one probability for tempo speed up (pn(1)),
                % and one for tempo slowing down pn(2)
                n_r_trans = zeros(R * N, N);
                for ri = 1:R
                    n_const = diag(ones(N, 1) * (1-sum(obj.pn)), 0);
                    % Also adjust for values at minN and maxN
                    if max_tempo_ss(ri) > min_tempo_ss(ri)
                        n_const(min_tempo_ss(ri), min_tempo_ss(ri)) = ...
                            1 - obj.pn(1);
                        n_const(max_tempo_ss(ri),max_tempo_ss(ri)) = ...
                            1 - obj.pn(2);
                    elseif max_tempo_ss(ri) == min_tempo_ss(ri)
                        n_const(min_tempo_ss(ri), min_tempo_ss(ri)) = 1;
                    end
                    n_up = diag(ones(N, 1) * obj.pn(1), 1);
                    n_down = diag(ones(N, 1) * obj.pn(2), -1);
                    % add the three matrices
                    temp = n_const + n_up(1:N, 1:N) + ...
                        n_down(1:N, 1:N);
                    % remove tempo change transitions below minN
                    temp(1:min_tempo_ss(ri)-1, :) = 0;
                    temp(:, 1:min_tempo_ss(ri)-1) = 0;
                    % remove tempo change transitions above maxN
                    temp(max_tempo_ss(ri)+1:N, :) = 0;
                    temp(:, max_tempo_ss(ri)+1:N) = 0;
                    n_r_trans((ri-1) * N + 1:ri * N, :) = temp;
                end
            else
                error('p_n has wrong dimensions!\n');
            end
            p = 1;
            % memory allocation:
            ri = zeros(num_state_ids * 3, 1);
            cj = zeros(num_state_ids * 3, 1);
            val = zeros(num_state_ids * 3, 1);
%             state_start_id = 1;
            for rhi = 1:R
                mi=1:M_from_pattern(rhi);
                for ni = min_tempo_ss(rhi):max_tempo_ss(rhi)
                    % decode m, n, r into state index i
                    lin_idx = sub2ind([M, N, R], mi, ...
                        repmat(ni, 1, M_from_pattern(rhi)), ...
                        repmat(rhi, 1, M_from_pattern(rhi)));
                    state_ids = state_from_substate(lin_idx);
                    % position of state j
                    mj = mod(mi + ni - 1, M_from_pattern(rhi)) + 1; 
                    % ----------------------------------------------------------
                    bar_crossing = mj < mi;
                    n_bc = sum(bar_crossing);
                    % number of non-bar-crossings
                    nn_bc = length(bar_crossing) - n_bc;
                    % possible transitions: 3 x T x R
                    ind_rn = (rhi - 1) * N + ni;
                    for n_ind = 1:3
                        if n_ind == 1 % tempo decrease
                            if ni == min_tempo_ss(rhi), continue; end
                            nj = ni - 1;
                        elseif n_ind == 2 % tempo constant
                            nj = ni;
                        else  % tempo increase
                            if ni == max_tempo_ss(rhi), continue; end
                            nj = ni + 1;
                        end
                        prob = n_r_trans(ind_rn, nj);
                        for rhj=1:R
                            prob2 = obj.pr(rhi, rhj);
                            lin_idx = sub2ind([M, N, R], mj(bar_crossing), ...
                                repmat(nj, 1, n_bc), ...
                                repmat(rhj, 1, n_bc));
                            j = state_from_substate(lin_idx);
                            ri(p:p+n_bc-1) = state_ids(bar_crossing);
                            cj(p:p+n_bc-1) = j;
                            val(p:p+n_bc-1) = prob * prob2 * (1-obj.p2s);
                            p = p + n_bc;
                        end
                    end
                    if obj.state_space.use_silence_state
                        % transition to silence state possible at bar
                        % transition
                        ri(p:p+n_bc-1) = state_ids(bar_crossing);
                        cj(p:p+n_bc-1) = silence_state_id;
                        val(p:p+n_bc-1) = obj.p2s;
                        p = p + n_bc;
                    end
                    % ----------------------------------------------------------
                    % inside the bar, no rhythm change possible
                    rhj = rhi;
                    % possible transitions: 3
                    for n_ind = 1:3 % decrease, constant, increase
                        if n_ind == 1 % tempo decrease
                            if ni == min_tempo_ss(rhi), continue; end
                            nj = ni - 1;
                        elseif n_ind == 2 % tempo constant
                            nj = ni;
                        else  % tempo increase
                            if ni == max_tempo_ss(rhi), continue; end
                            nj = ni + 1;
                        end
                        prob = n_r_trans((rhi-1)*N + ni, nj);
                        lin_idx = sub2ind([M, N, R], mj(~bar_crossing), ...
                                repmat(nj, 1, nn_bc), ...
                                repmat(rhj, 1, nn_bc));
                        j = state_from_substate(lin_idx);
                        ri(p:p+nn_bc-1) = state_ids(~bar_crossing);
                        cj(p:p+nn_bc-1) = j;
                        val(p:p+nn_bc-1) = prob;
                        p = p + nn_bc;
                    end
                end
            end
            % --------------------------------------------------------------
            if obj.state_space.use_silence_state
                p0 = p;
                % self transition
                cj(p) = silence_state_id;
                val(p) = 1 - obj.pfs;
                p = p + 1;
                % transition from silence state to m=1, n(:), r(:)
                n = [];
                r = [];
                error(['WARNING: silence state with old models is not', ...
                    'implemented yet!']);
                for i_r=1:R
                    n = [n, min_tempo_ss(i_r):max_tempo_ss(i_r)];
                    r = [r, ones(1, length(min_tempo_ss(i_r):max_tempo_ss(i_r))) ...
                        * i_r];
                end
                cj(p:p+length(n(:))-1) = ones(length(n(:)), 1) + ...
                    (n(:)-1)*M + (r(:)-1)*M*N;
                val(p:p+length(n(:))-1) = obj.pfs/(length(n(:)));
                p = p + length(n(:));
                ri(p0:p-1) = silence_state_id;
                idx = 1:p-1;
                % remove transitions to unpossible states (state = -1)
                idx = (cj(idx) > 0) & (ri(idx) > 0);
                obj.A = sparse(ri(idx), cj(idx), val(idx), ...
                    num_state_ids + 1, num_state_ids + 1);
            else
                idx = 1:p-1;
                % remove transitions to unpossible states (state = -1)
                idx = (cj(idx) > 0) & (ri(idx) > 0) & (val(idx) > 0);
                obj.A = sparse(ri(idx), cj(idx), val(idx), ...
                    num_state_ids, num_state_ids);
            end
            obj.n_transitions = length(find(ri(idx)));
        end
    end
end

