classdef AppsTest < matlab.unittest.TestCase
    % ExamplesTest tests the examples in the examples folder
    properties
        result_folder = fullfile(pwd, 'results');
        audio_files = {fullfile(pwd, '..', ...
            'examples/data/audio/guitar_duple.flac'), fullfile(pwd, '..', ...
            'examples/data/audio/guitar_triple.flac')};
    end
    
    methods (Test)
        function test_mirex_2013(testCase)
            addpath(fullfile(pwd, '..', 'apps/mirex_2013'));
            % Beat tracking with a pre-trained HMM
            compute_beats_mirex_2013(testCase.audio_files{1}, ...
                testCase.result_folder);
            [~, fname, ~] = fileparts(testCase.audio_files{1});
            results = load(fullfile(testCase.result_folder, ...
                [fname, '.beats.txt']));
            % add all beats and compare to expected solution
            exp_sum_beats = 776.24;
            act_sum_beats = sum(results(:, 1));
            testCase.verifyLessThan(abs(act_sum_beats-exp_sum_beats), ...
                1e-3);
        end
        

    end
    
end