classdef ExamplesTest < matlab.unittest.TestCase
    % ExamplesTest tests the examples in the examples folder
    properties
        result_folder = fullfile(pwd, 'results')
        audio_files = {fullfile(pwd, '..', ...
            'examples/data/audio/guitar_duple.flac'), fullfile(pwd, '..', ...
            'examples/data/audio/guitar_triple.flac')}
        example_path = fullfile(pwd, '..', 'examples')
    end
    
    methods (Test)
        function testEx1(testCase)
            addpath(testCase.example_path)
            fprintf('\ntestEx1: \n');
            % Beat tracking with a pre-trained HMM
            Results = ex1_beat_tracking_with_pretrained_hmm(...
                testCase.audio_files{1}, testCase.result_folder);
            % add all beats and compare to expected solution
            exp_sum_beats = 777.48;
            act_sum_beats = sum(Results{1}(:, 1));
            testCase.verifyLessThan(abs(act_sum_beats-exp_sum_beats), 1e-3);
        end
        
        function testEx2(testCase)
            addpath(testCase.example_path)
            fprintf('\ntestEx2: \n');
            % Beat tracking with a pre-trained PF
            % Initialize the random number generator using a seed of 1 
            % to make the results in this example repeatable
            rng('default'); rng(1);
            Results = ex2_beat_tracking_with_pretrained_pf(...
                testCase.audio_files{1}, testCase.result_folder);
            % add all beats and compare to expected solution
            exp_sum_beats = 777.6201;
            act_sum_beats = sum(Results{1}(:, 1));
            testCase.verifyLessThan(abs(act_sum_beats-exp_sum_beats), 1e-3);
        end
        
        function testEx3(testCase)
            addpath(testCase.example_path)
            fprintf('\ntestEx3: \n');
            rng('default'); rng(1);
            % Train a HMM and test it
            Results = ex3_train_and_test_hmm(testCase.audio_files{1}, ...
                testCase.audio_files, testCase.result_folder);
            % add all beats and compare to expected solution
            exp_sum_beats = 776.72;
            act_sum_beats = sum(Results{1}(:, 1));
            testCase.verifyLessThan(abs(act_sum_beats-exp_sum_beats), 1e-3);
        end
        
        function testEx4(testCase)
            addpath(testCase.example_path)
            fprintf('\ntestEx4: \n');
            rng('default'); rng(1);
            % Train a HMM and test it
            Results = ex4_train_and_test_hmm(testCase.audio_files{1}, ...
                testCase.audio_files, testCase.result_folder);
            % add all beats and compare to expected solution
            exp_sum_beats = 776.72;
            act_sum_beats = sum(Results{1}(:, 1));
            testCase.verifyLessThan(abs(act_sum_beats-exp_sum_beats), 1e-3);
        end
        
        function testEx5(testCase)
            addpath(testCase.example_path)
            fprintf('\ntestEx5: \n');
            rng('default'); rng(1);
            % Train a HMM and test it
            Results = ex5_train_and_test_pf(testCase.audio_files{1}, ...
                testCase.audio_files, testCase.result_folder);
            % add all beats and compare to expected solution
            exp_sum_beats = 777.02;
            act_sum_beats = sum(Results{1}(:, 1));
            testCase.verifyLessThan(abs(act_sum_beats-exp_sum_beats), 1e-3);
        end

    end
    
end