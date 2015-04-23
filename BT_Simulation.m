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
[~, cmdout] = system('git rev-parse HEAD');
% go back to where we came from
system('cd -');
fprintf('Git SHA-1: %s\n', cmdout);
fprintf('Process ID: %i\n', feature('getpid'));
% create folder for results
if ~exist(['./results/', num2str(sim_id)], 'dir')
    system(['mkdir ./results/', num2str(sim_id)]);
end
% load parameters ...
if exist('params', 'var')
    if isstruct(params)
        % from struct
        sim = Simulation(params, sim_id);   
    else
        if exist(params, 'file')
            config_fln = params;
        else
            error('ERROR BT_Simulation.m: Config file %s not found\n', ...
                params);
        end
    end
else
   config_fln = 'config_bt.m';
end
if exist('config_fln', 'var')
    % load parameters from config file
    fprintf(['* Copy ', config_fln, ' to ', num2str(sim_id), '/config_bt_dir.m\n']);
    system(['cp ', config_fln, ' ./results/', num2str(sim_id), '/config_bt_dir.m']);
    sim = Simulation('config_bt_dir', sim_id, ['./results/', num2str(sim_id)]);
end
% start training
sim = sim.train_system();
fprintf('\n');
tic;
% start simulation
sim.do_sim;
sim = sim.set_comp_time(toc/60);
% save parameters to file
sim.save_params();
fprintf('Simulation finished in %.3f minutes\n', sim.Params.compTime);
runtime = sim.Params.compTime;
end % end BT_Simulation
