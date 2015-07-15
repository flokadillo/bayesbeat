function [runtime] = BT_Simulation(sim_id, params)
% computes beats, downbeats, meter and tempo from audio files
% [runtime] = BT_Simulation(sim_id, params)
%
% ------------------------------------------------------------------------
% INPUT parameters:
% simId             simulation id. the results are stored in a folder with
%                   the name "simId" into the "results" folder
% params            can be either a matlab structure with the parameters or
%                   a string with the filename of a config file which
%                   contains the parameters
%
% OUTPUT parameters:
% runtime           runtime (excluding training) in minutes
%
%
% 06.09.2012 by Florian Krebs
%
% changelog:
% 03.01.2013 modularize code into subfunctions
% 29.01.2015 added config file name input parameter
% ------------------------------------------------------------------------
bayes_beat_path = '~/diss/src/matlab/beat_tracking/bayes_beat';
% output hash of current git revision
% change to base path
system(['cd ', bayes_beat_path]);
% get hash
[~, git_sha1] = system('git rev-parse HEAD');
% go back to where we came from
system('cd -');
fprintf('Git SHA-1: %s\n', git_sha1);
fprintf('Process ID: %i\n', feature('getpid'));
% load parameters ...
if exist('params', 'var')
    if isstruct(params)
        % parameters are already given as struct
        Params = params;
    else
        % parameters are given in a file
        if exist(params, 'file')
            config_fln = params;
            [config_path, fname, ~] = fileparts(config_fln);
            addpath(config_path);
            Params = eval(fname);
        else
            error('ERROR BT_Simulation.m: Config file %s not found\n', ...
                params);
        end
    end
else
    error('Please specify parameters or a config file!\n');
end
% make results folder
Params.results_path = fullfile(Params.results_path, num2str(sim_id));
if ~exist(Params.results_path, 'dir')
    system(['mkdir ', Params.results_path]);
end
% copy config file to simulation dir
if ~isstruct(params) && exist(params, 'file')
    fprintf(['* Copy ', params, ' to ', fullfile(Params.results_path, ...
        'config.m\n')]);
    system(['cp ', params, ' ', fullfile(Params.results_path, 'config.m')]);
end
% create Simulation object
sim = Simulation(Params, sim_id);
% start training
sim = sim.train_system();
fprintf('\n');
tic;
% start simulation
sim.do_sim;
sim = sim.set_comp_time(toc/60);
sim = sim.set_git_sha1(git_sha1);
% save parameters to file
sim.save_params();
fprintf('Simulation finished in %.3f minutes\n', sim.Params.compTime);
runtime = sim.Params.compTime;
end % end BT_Simulation
