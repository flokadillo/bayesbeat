function [] = BT_Simulation(sim_id)
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
sim = Simulation('config_bt');
if nargin == 1
    sim = sim.set_up_results_dir(sim_id);
end

% output hash of current git revision
[~, cmdout] = system('git rev-parse HEAD');
fprintf('Git SHA-1: %s\n', cmdout);

sim = sim.train_system();

tic;
fprintf('\n');

sim.do_sim;

sim = sim.set_comp_time(toc/60);

sim.save_params();


% profile viewer

end % end BT_Simulation