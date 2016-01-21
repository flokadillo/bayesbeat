classdef HMMTest < matlab.unittest.TestCase
    % ExamplesTest tests the examples in the examples folder
    properties
        result_folder = fullfile(pwd, 'results')
        audio_files = {fullfile(pwd, '..', ...
            'examples/data/audio/guitar_duple.flac'), fullfile(pwd, '..', ...
            'examples/data/audio/guitar_triple.flac')}
        example_path = fullfile(pwd, '..', 'examples')
    end
    
    methods (Test)
        function testBeatTrackingTransitionModelHMM2015(testCase)
            addpath(testCase.example_path)
            fprintf('\ntestBeatTrackingTransitionModelHMM2015: \n');
            % Create state space
            State_space_params.n_tempi = nan;
            State_space_params.n_patterns = 3;
            State_space_params.max_positions = [0.75; 1; 1];
            min_tempo_bpm = [50; 70; 90];
            max_tempo_bpm = [100; 120; 140];
            n_beats_from_pattern = [3; 4; 4];
            meter_from_pattern = [3, 4; 4, 4; 4, 4];
            frame_length = 0.02;
            pattern_names = ''; use_silence_state = 0; store_proximity = 0;
            state_space = BeatTrackingStateSpaceHMM2015(State_space_params, ...
                min_tempo_bpm, max_tempo_bpm, n_beats_from_pattern, ...
                meter_from_pattern, frame_length, pattern_names, ...
                use_silence_state, store_proximity);
            % Create transition model
            transition_params.transition_lambda = 100;
            transition_params.pr = [0.8, 0.1, 0.1; 
                                    0.1, 0.7, 0.2; 
                                    0.1, 0.2, 0.8];
            tm = BeatTrackingTransitionModelHMM2015(state_space, ...
                transition_params);
            % state at the end of a bar, pattern transitions possible
            from_state = 4002; 
            to_states = find(tm.A(from_state, :));
            prob = tm.A(from_state, to_states);
            testCase.verifyEqual(sum(to_states), 258559);
            testCase.verifyLessThan(...
                full(sum(prob.*prob)) - 0.555795169311110, 1e-12);
            testCase.verifyEqual(tm.n_transitions, 13735);
            testCase.verifyEqual(tm.is_corrupt, 0);
        end
        
    end
    
end