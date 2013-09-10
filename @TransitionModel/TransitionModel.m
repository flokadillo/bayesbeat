classdef TransitionModel
    % Transition Model Class
    properties (SetAccess=private)
        A               % sparse transition matrix [nStates x nStates]
        M               % number of positions in a 4/4 bar
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
        function obj = TransitionModel(M, N, R, pn, pr, pt, rhythm2meter, minN, maxN)
            
            obj.M = M;
            obj.N = N;
            obj.R = R;
            obj.pn = pn;
            obj.pr = pr;
            obj.pt = pt;
            obj.rhythm2meter = rhythm2meter;     
            obj.minN = minN;
            obj.maxN = maxN;
            if maxN > N
                error('N should be %i instead of %i\n', maxN, N);
            end
            
            numstates = M*N*R;
            tic;
            
            p = 1;
            
            % memory allocation:
            ri = zeros(numstates*3,1); cj = zeros(numstates*3,1); val = zeros(numstates*3,1);
            
            % first state with n >= min_n
            start_i = sub2ind([M N R], 1, max([minN(1); 2]), 1);
            
            prob2 = zeros(R, 1);
            perc = round(numstates/15);
            fprintf('* Set up transition model .');
            
            for i = start_i:numstates     % at time k
                
                if rem(i, perc) == 0
                    fprintf('.');
                end
                
                % decode state number to m and n
                [mi, ni, rhi] = ind2sub([M, N, R], i);
                ti = rhythm2meter(rhi);
                % filter out states violating the tempo ...
                % constraints, states at the borders are defined separately at the
                % end
                if (ni <= minN(rhi)) || (ni >= maxN(rhi))
                    continue;
                end
                
                meter = (ti+2)/4;
                mj = mod(mi + ni - 1, round(M * meter)) + 1; % new position
                % --------------------------------------------------------------
                if mj < mi  % bar pointer crossed end of bar
                    % possible transitions: 3 x T x R
                    for n_ind = 1:3
                        if n_ind == 1 % tempo decrease
                            nj = ni - 1;
                            prob = pn/2;
                        elseif n_ind == 2 % tempo constant
                            nj = ni;
                            prob = 1-pn;
                        else  % tempo increase
                            nj = ni+1;
                            prob = pn/2;
                        end
                        
                        for rhj = 1:R
                            tj = rhythm2meter(rhj);
                            if tj == ti
                                if rhj == rhi % meter and rhythm constant
                                    prob2(rhj) = (1-pt) * (1-pr);
                                else % meter constant, rhythm change
                                    prob2(rhj) = (1-pt) * (pr/(R-1));
                                end
                            else
                                if rhj == rhi % meter change, rhythm constant
                                    prob2(rhj) = 0;
                                else % meter change, rhythm change
                                    prob2(rhj) = pt * (pr/(R-1));
                                end
                            end
                            j = sub2ind([M, N, R], mj, nj, rhj); % get new state index
                            ri(p) = i;  cj(p) = j;
                            p = p + 1;
                        end
                        % within a tempo state, all rhythmic patterns have to sum to
                        % one
                        if sum(prob2) == 0
                            val(p-R:p-1) = 0;
                        else
                            val(p-R:p-1) = prob .* (prob2 / sum(prob2));
                        end
                        
                    end
                    % --------------------------------------------------------------
                else % inside the bar
                    % possible transitions: 3
                    for n_ind = 1:3 % constant, decrease, increase
                        if n_ind == 1 % tempo decrease
                            nj = ni - 1;
                            prob = pn/2;
                        elseif n_ind == 2 % tempo constant
                            nj = ni;
                            prob = 1-pn;
                        else  % tempo increase
                            nj = ni+1;
                            prob = pn/2;
                        end
                        j = sub2ind([M, N, R], mj, nj, rhi);
                        ri(p) = i;  cj(p) = j;   val(p) = prob;
                        p = p + 1;
                    end
                end
                
            end
            
            % --------------------------------------------------------------
            % set probabilities for states with min and max tempo (borders of state space)
            for mi=1:M  % position
                for rhi=1:R  % rhythmic patterns
                    meter = (rhythm2meter(rhi)+2)/4;
                    % -----------------------------------------------
                    % ni = minimal tempo:
                    % -----------------------------------------------
                    ni = minN(rhi);
                    % only 2 possible transitions
                    % 1) tempo constant
                    nj = ni;
                    mj = mod(mi + ni - 1, round(M * meter)) + 1; % new position
                    i = sub2ind([M N R], mi, ni, rhi); % state i
                    j = sub2ind([M N R], mj, nj, rhi); % state j
                    ri(p) = i;  cj(p) = j;   val(p) = 1-pn;   p = p + 1;
                    % 2) tempo increase
                    nj = ni + 1;
                    j = sub2ind([M N R], mj, nj, rhi);
                    ri(p) = i;  cj(p) = j;   val(p) = pn;   p = p + 1;
                    % -----------------------------------------------
                    % ni = maximal tempo:
                    % -----------------------------------------------
                    ni = min([maxN(rhi), N]);
                    % only 2 possible transitions
                    % 1) tempo constant
                    nj = ni;
                    mj = mod(mi + ni - 1, round(M * meter)) + 1; % new position
                    i = sub2ind([M N R], mi, ni, rhi);
                    j = sub2ind([M N R], mj, nj, rhi);
                    ri(p) = i;  cj(p) = j;   val(p) = 1-pn;   p = p + 1;
                    % 2) tempo decrease
                    nj = ni - 1;
                    j = sub2ind([M N R],mj,nj,rhi);
                    ri(p) = i;  cj(p) = j;   val(p) = pn;   p = p + 1;
                end
            end
            
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
            zero_probs_j = abs(sum_over_j) < 1e-10;  % p ≃ 0
            one_probs_j = abs(sum_over_j-1) < 1e-10; % p ≃ 1
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
                    fprintf('      sum: p=%.3f\n', sumProb);
                end

            end
            
        end
        
        
    end
end
