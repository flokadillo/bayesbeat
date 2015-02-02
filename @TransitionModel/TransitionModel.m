classdef TransitionModel
    % Transition Model Class
    properties (SetAccess=public)
        tempo_transition_probs % [N*R x N]
    end
    
    properties (SetAccess=private)
        A               % sparse transition matrix [nStates x nStates]
        M               % number of max bar positions
        Meff            % number of positions for each meter
        N               % number of tempo states
        R               % number of rhythmic pattern states
        pn              % probability of a switch in tempo
        pr              % probability of a switch in rhythmic pattern
        rhythm2meter_state    % assigns each rhythmic pattern to a meter state
        minN            % min tempo (n_min) for each rhythmic pattern
        maxN            % max tempo (n_max) for each rhythmic pattern
        state_type      % 'discrete' or 'continuous'
        evaluate_fh
        sample_fh
        use_silence_state   % [true/false] use silence stateto pause tracker
        p2s                 % prior probability to go into silence state
        pfs                 % prior probability to exit silence state
        tempo_state_map     % [n_states, 1] contains for each state the
        %                     corresponding tempo or Nan
        position_state_map  % [n_states, 1] contains for each state the
        %                     corresponding position or Nan
        rhythm_state_map    % [n_states, 1] contains for each state the
        %                     corresponding rhythm or Nan
        
    end
    
    
    
    methods
        function obj = TransitionModel(M, Meff, N, R, pn, pr, rhythm2meter_state, ...
                minN, maxN, use_silence_state, p2s, pfs)
            % save parameters
            obj.M = M;
            obj.Meff = Meff;
            obj.N = N;
            obj.R = R;
            obj.pn = pn;
            obj.pr = pr;
            obj.rhythm2meter_state = rhythm2meter_state;
            obj.minN = minN;
            obj.maxN = maxN;
            
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
            % check if N is big enough to cover given tempo range
            if max(maxN) > N
                error('N should be %i instead of %i\n', max(maxN), N);
            end
            fprintf('* Set up transition model .');
            obj = obj.make_whiteleys_tm(1);
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
            numstates = obj.M * obj.N * obj.R;
            % alloc memory for
            obj.tempo_state_map = nan(numstates, 1);
            obj.position_state_map = nan(numstates, 1);
            obj.rhythm_state_map = nan(numstates, 1);
            if (size(obj.pr, 1) == obj.R) && (obj.R > 1)
                % pr is a matrix RxR (R>1), do nothing
            elseif size(obj.pr, 1) == 1
                pr_mat = ones(obj.R, obj.R) * (obj.pr / (obj.R-1));
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
            ri = zeros(numstates*3,1); cj = zeros(numstates*3,1); val = zeros(numstates*3,1);
            % first state with n >= min_n
%             start_i = sub2ind([M N R], 1, max([minN(1); 2]), 1);
%             prob2 = zeros(R, 1);
           
            for rhi = 1:obj.R
                if do_output, fprintf('.'); end;
                mi=1:obj.Meff(obj.rhythm2meter_state(rhi));
                ti = obj.rhythm2meter_state(rhi);
                for ni = obj.minN(rhi)+1:obj.maxN(rhi)-1
                    % decode m, n, r into state index i
                    i = sub2ind([obj.M, obj.N, obj.R], mi, repmat(ni, 1, ...
                        obj.Meff(obj.rhythm2meter_state(rhi))), ...
                        repmat(rhi, 1, obj.Meff(obj.rhythm2meter_state(rhi))));
                    % save state mappings
                    obj.tempo_state_map(i) = ni;
                    obj.position_state_map(i) = mi;
                    obj.rhythm_state_map(i) = rhi;
                    % position of state j
                    mj = mod(mi + ni - 1, obj.Meff(ti)) + 1; % new position
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
                mj = mod(mi + ni - 1, obj.Meff(obj.rhythm2meter_state(rhi))) + 1; % new position
                i = j_r(rhi) + (ni - 1) * obj.M + mi;
                j = j_r(rhi) + (nj - 1) * obj.M + mj;
                ri(p:p+obj.M-1) = i;  cj(p:p+obj.M-1) = j;   
                val(p:p+obj.M-1) = n_r_trans((rhi-1)*obj.N + ni, nj);   
                p = p + obj.M;
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
%                 mj = mod(mi + ni - 1, obj.Meff(obj.rhythm2meter_state(rhi))) + 1; % new position
                i = j_r(rhi) + (ni-1)*obj.M + mi;
                ri(p:p+obj.M-1) = i;  cj(p:p+obj.M-1) = j;   
                val(p:p+obj.M-1) = n_r_trans((rhi-1)*obj.N + ni, nj);   
                p = p + obj.M;
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
                    numstates+1, numstates+1);
            else
                obj.A = sparse(ri(1:p-1), cj(1:p-1), val(1:p-1), ...
                    numstates, numstates);
            end
        end
        
        function obj = make_2015_tm(obj)
            % This function creates a transition function with the following
            % properties: each tempostate has a different number of position
            % states
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
                fprintf('    Number of non-zero states: %i (%.1f %%)\n', sum(sum_over_j==1), sum(sum_over_j==1)*100/size(obj.A,1));
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
                    [m, n, r] =  ind2sub([obj.M obj.M obj.R], corrupt_states_i(1));
                    fprintf('    State %i (%i - %i - %i) has transition TO: \n',corrupt_states_i(1), m, n, r);
                    trans_states = find(obj.A(corrupt_states_i(1),:));
                    sumProb = 0;
                    for i=1:length(trans_states)
                        [m, n, r] =  ind2sub([obj.M obj.M obj.R], trans_states(i));
                        fprintf('      %i (%i - %i - %i) with p=%.3f\n', trans_states(i), m, n, r, full(obj.A(corrupt_states_i(1), trans_states(i))));
                        sumProb = sumProb + full(obj.A(corrupt_states_i(1),trans_states(i)));
                    end
                    fprintf('      sum: p=%.10f\n', sumProb);
                end
                
            end
            
        end
        
        
    end
end
