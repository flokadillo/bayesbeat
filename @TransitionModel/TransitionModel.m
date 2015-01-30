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
        use_silence_state
        tempo_state_map     % [n_states, 1] contains for each state the 
        %                     corresponding tempo or Nan
        position_state_map  % [n_states, 1] contains for each state the 
        %                     corresponding position or Nan
        rhythm_state_map    % [n_states, 1] contains for each state the 
        %                     corresponding rhythm or Nan
        
    end
    
    
    
    methods
        function obj = TransitionModel(M, Meff, N, R, pn, pr, rhythm2meter_state, minN, maxN, use_silence_state, p2s, pfs)
            obj.M = M;
            obj.Meff = Meff;
            obj.N = N;
            obj.R = R;
            obj.pn = pn;
            obj.pr = pr;
            obj.rhythm2meter_state = rhythm2meter_state;
            obj.minN = minN;
            obj.maxN = maxN;
            numstates = M*N*R;
            obj.tempo_state_map = nan(numstates, 1);
            obj.position_state_map = nan(numstates, 1);
            obj.rhythm_state_map = nan(numstates, 1);
            if nargin < 11
                use_silence_state = 0;
                p2s = 0;
                pfs = 0;
            end
            if use_silence_state
               silence_state_id = obj.M * obj.N *obj.R + 1;
            else
                p2s = 0;
                pfs = 0;
            end
            if max(maxN) > N
                error('N should be %i instead of %i\n', max(maxN), N);
            end
            % set up pattern transition matrix
            if (size(obj.pr, 1) == obj.R) && (obj.R > 1)
                % pr is a matrix RxR (R>1), do nothing
            elseif size(obj.pr, 1) == 1
                obj.pr = ones(obj.R, obj.R) * (obj.pr / (obj.R-1));
                obj.pr(logical(eye(obj.R))) = (1-pr);
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
            start_i = sub2ind([M N R], 1, max([minN(1); 2]), 1);
            prob2 = zeros(R, 1);
            perc = round(numstates/15);
            fprintf('* Set up transition model .');
            for rhi = 1:R
                fprintf('.');
                mi=1:obj.Meff(rhythm2meter_state(rhi));
                ti = rhythm2meter_state(rhi);
                for ni = minN(rhi)+1:maxN(rhi)-1
                    
                    % decode m, n, r into state index i
                    i = sub2ind([M, N, R], mi, repmat(ni, 1, ...
                        obj.Meff(rhythm2meter_state(rhi))), ...
                        repmat(rhi, 1, obj.Meff(rhythm2meter_state(rhi))));
                    % save state mappings
                    obj.tempo_state_map(i) = ni;
                    obj.position_state_map(i) = mi;
                    obj.rhythm_state_map(i) = rhi;
                    
                    mj = mod(mi + ni - 1, obj.Meff(ti)) + 1; % new position
                    % --------------------------------------------------------------
                    bar_crossing = mj < mi;
                    n_bc = sum(bar_crossing);
                    nn_bc = length(bar_crossing) - n_bc;
                    % possible transitions: 3 x T x R
                    ind_rn = (rhi-1)*N + ni;
                    
                    for n_ind = 1:3
                        if n_ind == 1 % tempo decrease
                            nj = ni - 1;
                            j_n = mj(bar_crossing) + (nj-1)*M;
                        elseif n_ind == 2 % tempo constant
                            nj = ni;
                            j_n = mj(bar_crossing) + (nj-1)*M;
                        else  % tempo increase
                            nj = ni+1;
                            j_n = mj(bar_crossing) + (nj-1)*M;
                        end
                        prob = n_r_trans(ind_rn, nj);
                        for rhj=1:R
                            prob2 = obj.pr(rhi, rhj);
                            j = (rhj-1)*N*M + j_n;
                            %                             j = sub2ind([M, N, R], mj, nj, rhj); % get new state index
                            
                            ri(p:p+n_bc-1) = i(bar_crossing);
                            cj(p:p+n_bc-1) = j;
                            val(p:p+n_bc-1) = prob * prob2 * (1-p2s);
                            p = p + n_bc;
                            %                         end
                            % within a tempo state, all rhythmic patterns have to sum to
                            % one
                            %                         if sum(prob2) == 0
                            %                             val(p-R:p-1) = 0;
                            %                         else
                            
                        end
                    end
                    if use_silence_state
                        % transition to silence state possible at bar
                        % transition
                        ri(p:p+n_bc-1) = i(bar_crossing);
                        cj(p:p+n_bc-1) = silence_state_id;
                        val(p:p+n_bc-1) = p2s;
                        p = p + n_bc;
                    end
                    % --------------------------------------------------------------
                     % inside the bar
                    j_mr = (rhi-1)*N*M + mj(~bar_crossing);
                    % possible transitions: 3
                    %                         nj = [ni - 1, ni, ni + 1];
                    for n_ind = 1:3 % decrease, constant, increase
                        if n_ind == 1 % tempo decrease
                            nj = ni - 1;
                        elseif n_ind == 2 % tempo constant
                            nj = ni;
                        else  % tempo increase
                            nj = ni+1;
                        end
                        prob = n_r_trans((rhi-1)*obj.N + ni, nj);
                        j = (nj-1)*M + j_mr;
                        %                         j = sub2ind([M, N, R], mj, nj, rhi);
                        ri(p:p+nn_bc-1) = i(~bar_crossing);
                        cj(p:p+nn_bc-1) = j;
                        val(p:p+nn_bc-1) = prob;
                        p = p + nn_bc;
                    end
                    %                     end
                end
            end
            
            % --------------------------------------------------------------
            % set probabilities for states with min and max tempo (borders of state space)
            j_r = (0:R-1)*N*M;
            %             for mi=1:M  % position
            mi = 1:M;
            for rhi=1:R
                %                 rhi = 1:R;
                %                 for rhi=1:R  % rhythmic patterns
                
                % -----------------------------------------------
                % ni = minimal tempo:
                % -----------------------------------------------
                ni = minN(rhi);
                % only 2 possible transitions
                % 1) tempo constant
                nj = ni;
                mj = mod(mi + ni - 1, obj.Meff(rhythm2meter_state(rhi))) + 1; % new position
                i = j_r(rhi) + (ni-1)*M + mi;
                j = j_r(rhi) + (nj-1)*M + mj;
                %                     i = sub2ind([M N R], mi, ni, rhi); % state i
                %                     j = sub2ind([M N R], mj, nj, rhi); % state j
                ri(p:p+M-1) = i;  cj(p:p+M-1) = j;   val(p:p+M-1) = n_r_trans((rhi-1)*obj.N + ni, nj);   p = p + M;
                % 2) tempo increase
                %                     nj = ni + 1;
                %                     j = sub2ind([M N R], mj, nj, rhi);
                j = j + M;
                ri(p:p+M-1) = i;  cj(p:p+M-1) = j;   val(p:p+M-1) = 1 - val(p-M:p-1);   p = p + M;
                % -----------------------------------------------
                % ni = maximal tempo:
                % -----------------------------------------------
                ni = min([maxN(rhi), N]);
                % only 2 possible transitions
                % 1) tempo constant
                nj = ni;
                mj = mod(mi + ni - 1, obj.Meff(rhythm2meter_state(rhi))) + 1; % new position
                i = j_r(rhi) + (ni-1)*M + mi;
                j = j_r(rhi) + (nj-1)*M + mj;
                %                     i = sub2ind([M N R], mi, ni, rhi);
                %                     j = sub2ind([M N R], mj, nj, rhi);
                ri(p:p+M-1) = i;  cj(p:p+M-1) = j;   val(p:p+M-1) = n_r_trans((rhi-1)*obj.N + ni, nj);   p = p + M;
                % 2) tempo decrease
                %                     nj = ni - 1;
                %                     j = sub2ind([M N R],mj,nj,rhi);
                j = j - M;
                ri(p:p+M-1) = i;  cj(p:p+M-1) = j;   val(p:p+M-1) = 1 - val(p-M:p-1);   p = p + M;
            end
            
            if use_silence_state
                p0 = p;
                % self transition
                cj(p) = silence_state_id;
                val(p) = 1 - pfs; 
                p = p + 1;
                % transition from silence state to m=1, n(:), r(:)
                n = [];
                r = [];
                for i_r=1:R
                    n = [n, minN(i_r):maxN(i_r)];
                    r = [r, ones(1, length(minN(i_r):maxN(i_r))) * i_r];
                end
                cj(p:p+length(n(:))-1) = ones(length(n(:)), 1) + (n(:)-1)*M + (r(:)-1)*M*N;
                val(p:p+length(n(:))-1) = pfs/(length(n(:)));
                p = p + length(n(:));
                ri(p0:p-1) = silence_state_id;
                obj.A = sparse(ri(1:p-1), cj(1:p-1), val(1:p-1), numstates+1, numstates+1);
            else
                obj.A = sparse(ri(1:p-1), cj(1:p-1), val(1:p-1), numstates, numstates);
            end
            %             end
            
            fprintf(' done\n');
             
            
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
