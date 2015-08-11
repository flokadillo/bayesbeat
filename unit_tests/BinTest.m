classdef BinTest < matlab.unittest.TestCase
    % ExamplesTest tests the examples in the examples folder
    properties
        result_folder = fullfile(pwd, 'temp');
        audio_files = {fullfile(pwd, '..', ...
            'examples/data/audio/guitar_duple.flac'), fullfile(pwd, '..', ...
            'examples/data/audio/guitar_triple.flac')};
    end
    
    methods (Test)
        function test_mirex_2013(testCase)
            addpath(fullfile(pwd, '..', 'bin/mirex_2013'));
            % Beat tracking with a pre-trained HMM
            results_fln = fullfile(testCase.result_folder, ...
                'test.beats.txt');
            compute_beats_mirex_2013(testCase.audio_files{1}, results_fln);
            results = load(results_fln);
            % add all beats and compare to expected solution
            exp_sum_beats = 777.36;
            act_sum_beats = sum(results);
            testCase.verifyLessThan(abs(act_sum_beats-exp_sum_beats), ...
                1e-3);
        end
        

    end
    
end