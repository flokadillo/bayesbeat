function [runtime] = BT_Simulation(sim_id, params)
% computes beats, downbeats, meter and tempo from audio files
% [] = Simulation()
%
% ------------------------------------------------------------------------
% INPUT parameters:
% simId
%
% OUTPUT parameters:
% none, results are saved to database
%
%
% 06.09.2012 by Florian Krebs
%
% changelog:
% 03.01.2013 modularize code into subfunctions
% ------------------------------------------------------------------------
addpath('~/diss/src/matlab/beat_tracking/SilverBeat/utils')
% profile on

% output hash of current git revision
[~, cmdout] = system('git rev-parse HEAD');
fprintf('Git SHA-1: %s\n', cmdout);
fprintf('Process ID: %i\n', feature('getpid'));

if ~exist(['./results/', num2str(sim_id)], 'dir')
    system(['mkdir ./results/', num2str(sim_id)]);
end
    
if exist('params', 'var')
   sim = Simulation(params, sim_id);   
else
   fprintf('* Copy config_bt.m to %s\n', [num2str(sim_id), '/config_bt_dir.m']);
   system(['cp ./config_bt.m ./results/', num2str(sim_id), '/config_bt_dir.m']);
   sim = Simulation('config_bt_dir', sim_id, ['./results/', num2str(sim_id)]);
end

sim = sim.train_system();

tic;
fprintf('\n');

sim.do_sim;

sim = sim.set_comp_time(toc/60);

sim.save_params();

fprintf('Simulation finished in %.3f minutes\n', sim.Params.compTime);
% profile viewer
runtime = sim.Params.compTime;
end % end BT_Simulation
