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
        pt              % probability of a switch in meter
        rhythm2meter    % assigns each rhythmic pattern to a meter
        minN            % min tempo (n_min) for each rhythmic pattern
        maxN            % max tempo (n_max) for each rhythmic pattern
        state_type      % 'discrete' or 'continuous'
        evaluate_fh
        sample_fh
        
        
    end
    
    
    
    methods
        function obj = TransitionModel(M, Meff, N, R, pn, pr, pt, rhythm2meter, minN, maxN)
            obj.M = M;
            obj.Meff = Meff;
            obj.N = N;
            obj.R = R;
            obj.pn = pn;
            obj.pr = pr;
            obj.pt = pt;
            obj.rhythm2meter = rhythm2meter;
            obj.minN = minN;
            obj.maxN = maxN;
            if max(maxN) > N
                error('N should be %i instead of %i\n', max(maxN), N);
            end
            % set up pattern transition matrix
            if size(obj.pr, 1) == 1
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
            numstates = M*N*R;
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
                mi=1:obj.Meff(rhythm2meter(rhi));
                ti = rhythm2meter(rhi);
                for ni = minN(rhi):N
                    %             for i = start_i:numstates     % at time k
%                     if rem(i, perc) == 0
%                         fprintf('.');
%                     end
                    % decode state number to m and n
                    i = sub2ind([M, N, R], mi, repmat(ni, 1, obj.Meff(rhythm2meter(rhi))), ...
                        repmat(rhi, 1, obj.Meff(rhythm2meter(rhi))));
                    %                     [mi, ni, rhi] = ind2sub([M, N, R], i);
                    
                    % filter out states violating the tempo ...
                    % constraints, states at the borders are defined separately at the
                    % end
                    if (ni <= minN(rhi)) || (ni >= maxN(rhi))
                        continue;
                    end
                    
                    mj = mod(mi + ni - 1, obj.Meff(ti)) + 1; % new position
                    % --------------------------------------------------------------
                    bar_crossing = mj < mi;
                    n_bc = sum(bar_crossing);
                    nn_bc = length(bar_crossing) - n_bc;
                    %                     if mj < mi  % bar pointer crossed end of bar
                    % possible transitions: 3 x T x R
                    ind_rn = (rhi-1)*N + ni;
                    
                    for n_ind = 1:3
                        if n_ind == 1 % tempo decrease
                            nj = ni - 1;
                            j_n = mj(bar_crossing) + (ni-2)*M;
                        elseif n_ind == 2 % tempo constant
                            j_n = mj(bar_crossing) + (ni-1)*M;
                            nj = ni;
                        else  % tempo increase
                            j_n = mj(bar_crossing) + ni*M;
                            nj = ni+1;
                        end
                        prob = n_r_trans(ind_rn, nj);
                        for rhj=1:R
                            prob2 = obj.pr(rhi, rhj);
                            j = (rhj-1)*N*M + j_n;
                            %                             j = sub2ind([M, N, R], mj, nj, rhj); % get new state index
                            
                            ri(p:p+n_bc-1) = i(bar_crossing);
                            cj(p:p+n_bc-1) = j;
                            val(p:p+n_bc-1) = prob * prob2;
                            p = p + n_bc;
                            %                         end
                            % within a tempo state, all rhythmic patterns have to sum to
                            % one
                            %                         if sum(prob2) == 0
                            %                             val(p-R:p-1) = 0;
                            %                         else
                            
                        end
                    end
                    % --------------------------------------------------------------
                    %                     else % inside the bar
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
                mj = mod(mi + ni - 1, obj.Meff(rhythm2meter(rhi))) + 1; % new position
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
                mj = mod(mi + ni - 1, obj.Meff(rhythm2meter(rhi))) + 1; % new position
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
            %             end
            
            fprintf(' done\n');
            
            ri = ri(1:p-1); cj = cj(1:p-1); val = val(1:p-1);
            obj.A = sparse(ri, cj, val, numstates, numstates);
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
