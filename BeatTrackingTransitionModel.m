classdef BeatTrackingTransitionModel < handle
    % Transition model base class to be used for Hidden Markov Models
    
    properties
        A                       % sparse transition matrix [nStates x nStates]
        state_space             % state_space object
        pr                      % rhythmic pattern transition matrix
        minN                    % min tempo (n_min) for each rhythmic 
        %                           pattern [R x 1]
        maxN                    % max tempo (n_max) for each rhythmic pattern 
        %                           [R x 1]
        p2s                     % prior probability to go into silence state
        pfs                     % prior probability to exit silence state
        n_transitions           % number of transitions
    end
    
    methods
        function obj = BeatTrackingTransitionModel(state_space, ...
            transition_params)
            obj.state_space = state_space;
            obj.pr = transition_params.pr;
            if obj.state_space.use_silence_state
                obj.pfs = transition_params.pfs;
                obj.p2s = transition_params.p2s;
            end
        end
    end   
end

