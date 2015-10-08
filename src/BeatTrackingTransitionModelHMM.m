classdef BeatTrackingTransitionModelHMM < handle
    % BeatTrackingTransitionModelHMM 
    % Transition model base class to be used for Hidden Markov Models
    
    properties
        A                       % sparse transition matrix [nStates x nStates]
        state_space             % state_space object
        pr                      % rhythmic pattern transition matrix
        p2s                     % prior probability to go into silence state
        pfs                     % prior probability to exit silence state
        n_transitions           % number of transitions
    end
    
    methods
        function obj = BeatTrackingTransitionModelHMM(state_space, ...
                transition_params)
            obj.state_space = state_space;
            obj.pr = transition_params.pr;
            if obj.state_space.use_silence_state
                obj.pfs = transition_params.pfs;
                obj.p2s = transition_params.p2s;
            else
                obj.pfs = 0;
                obj.p2s = 0;
            end
        end
        
        function error = is_corrupt(obj, dooutput)
            if nargin == 1, dooutput = 0; end
            error = 1;
            if dooutput, fprintf('*   Checking the Transition Matrix ... '); end
            % sum over columns j
            sum_over_j = full(sum(obj.A, 2));
            % find corrupt states: sum_over_j should be either 0 (if state is
            % never visited) or 1
            zero_probs_j = abs(sum_over_j) < 1e-2;  % p ≃ 0
            one_probs_j = abs(sum_over_j - 1) < 1e-2; % p ≃ 1
            corrupt_states_i = find(~zero_probs_j & ~one_probs_j);
            if dooutput
                fprintf('    Number of non-zero states: %i (%.1f %%)\n', ...
                    sum(sum_over_j==1), sum(sum_over_j==1)*100/size(obj.A,1));
                memory = whos('obj.A');
                fprintf('    Memory: %.3f MB\n',memory.bytes/1e6);
            end
            if isempty(corrupt_states_i)
                error = 0;
            else
                fprintf('    Number of corrupt states: %i  ', ...
                    length(corrupt_states_i));
                fprintf('    Example:\n');
                [position, tempo, pattern] = obj.state_space.decode_state(...
                    corrupt_states_i(1));
                fprintf('      State %i (%.3f - %.5f - %i) has transition to:\n', ...
                    corrupt_states_i(1), position, tempo, pattern);
                to_states = find(obj.A(corrupt_states_i(1), :));
                for i=1:length(to_states)
                    [position, tempo, pattern] = obj.state_space.decode_state(...
                        to_states(i));
                    fprintf('        %i (%.3f - %.5f - %i) with p=%.10f\n', ...
                        to_states(i), position, tempo, pattern, ...
                        full(obj.A(corrupt_states_i(1), to_states(i))));
                end
                sumProb = sum(full(obj.A(corrupt_states_i(1), to_states(:))));
                fprintf('      sum: p=%.10f\n', sumProb);
            end
        end 
        
    end
end

