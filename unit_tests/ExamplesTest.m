classdef ExamplesTest < matlab.unittest.TestCase
    % ExamplesTest tests the examples in the examples folder
    properties
        test_audio = fullfile(pwd, 'data', 'train1.flac');
        result_folder = fullfile(pwd, 'temp');
        hmm_model = fullfile(pwd, 'models', 'hmm_boeck.mat');
        pf_model = fullfile(pwd, 'models', 'pf_boeck.mat');
    end
    
    methods (Test)
        function testEx1(testCase)
            Results = ex1_beat_tracking_with_pretrained_hmm(...
                testCase.test_audio, testCase.result_folder, ...
                testCase.hmm_model);
            % add all beats and compare to expected solution
            exp_sum_beats = 40.68;
            act_sum_beats = sum(Results{1}(:, 1));
            testCase.verifyEqual(act_sum_beats, exp_sum_beats);
        end
        
        function testEx2(testCase)
            % Initialize the random number generator using a seed of 1 
            % to make the results in this example repeatable
            rng('default');
            rng(1);
            Results = ex2_beat_tracking_with_pretrained_pf(...
                testCase.test_audio, testCase.result_folder, ...
                testCase.pf_model);
            % add all beats and compare to expected solution
            exp_sum_beats = 37.52;
            act_sum_beats = sum(Results{1}(:, 1));
            testCase.verifyEqual(act_sum_beats, exp_sum_beats);
        end

    end
    
end