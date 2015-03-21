function Params = PF_config(base_path)
% [Params] = config_bt
%   specifies parameters for beat tracking algorithm
% ----------------------------------------------------------------------
% INPUT Parameter:
%   none
%
% OUTPUT Parameter:
%   Params            : structure array with beat tracking parameters
%
% 06.09.2012 by Florian Krebs
% ----------------------------------------------------------------------

% system name
Params.system = 'PF_MeterTracker';
Params.dataset = 'CMCMDa_small';

% Path settings
if exist('base_path', 'var')
    Params.base_path = base_path;
else
    Params.base_path = '/homedtic/amurthy/UPFWork_Server/PhD';
end
Params.data_path = fullfile(Params.base_path, 'Data', Params.dataset);
Params.results_path = fullfile(Params.base_path, 'BayesResults');
Params.temp_path = fullfile(Params.base_path, 'bayesbeat', 'temp');

% SIMULATION PARAMETERS:
% ======================

% If n_depends_on_r=true, then use different tempo limits for each rhythm
% state
Params.n_depends_on_r = 1;
% If save_inference_data=true, then save complete posterior probability to
% file. This is useful for visualisations.
Params.save_inference_data = 0;
% If reorganize_bars_into_cluster=true, then reorganise features into
% patterns as given by the cluster_assignment_file. Otherwise, Data.extract_feats_per_file_pattern_barPos_dim 
%is loaded from file.
Params.reorganize_bars_into_cluster = 0; % reorganize in Data.extract_feats_per_file_pattern_barPos_dim
% Inference and model settings {'HMM_viterbi', 'HMM_forward', 'PF',
% 'PF_viterbi'}
Params.inferenceMethod = 'PF';

% SYSTEM PARAMETERS:
% ==================

% State space size
% ----------------

% Maximum position state (used for the meter with the longest duration)
Params.M = 1600;
% Maximum tempo state 
Params.N = 16;
% Number of rhythmic pattern states
Params.R = 8;
% Meters that are modelled by the system, e.g., [9, 3; 8, 4] 
Params.meters = [3, 5, 7, 8; 4, 8, 8, 8];   % Make sure its in increasing order, bug in code otherwise!
Params.meterNames = {'rupaka', 'kChapu', 'mChapu', 'adi'};
% Number of position grid points per whole note. This is important for the
% observation model, as parameters are tied within this grid.
Params.whole_note_div = 64; 
% Number of grid points of one pattern per meter
% Params.barGrid_eff = Params.whole_note_div * (Params.meters(1, :) ./ Params.meters(2, :)); 
% Length of rhythmic patterns {beat', 'bar'}
Params.pattern_size = 'bar'; % 'beat' or 'bar'
% Audio frame length [sec]
Params.frame_length = 0.02;
% Model initial distribution over tempo states by mixture of init_n_gauss
% Gaussians.
Params.init_n_gauss = 0;
% Use one state to detect silence
Params.use_silence_state = 0;
% Probability of rhythmic pattern change
Params.pr = 0;

% PF parameters
% -------------

% Number of particles
Params.nParticles = 6000;
% Standard deviation of tempo transition. Note that the tempo n is normalised
% by dividing by M, so the actual sigma is sigmaN * M.
Params.sigmaN = 0.0001; 
% If the effective sample size is below ratio_Neff * nParticles, resampling is performed.
Params.ratio_Neff = 0.1;
Params.res_int = 30;
% Type of resampling scheme to be used:
%   0) Standard SISR (systematic resampling)
%   1) APF
%   2) Mixture PF using k-means clustering (MPF)
%   3) Auxiliary mixture particle filter (AMPF)
Params.resampling_scheme = 3;
Params.resampling_scheme_name = {'SISR', 'APF', 'MPF', 'AMPF'};
Params.store_name = Params.resampling_scheme_name{Params.resampling_scheme+1};
Params.do_viterbi_filtering = 0;

% APF parameters
% ..............
% Warping function of weights for APF and AMPF
Params.warp_fun = '@(x)x.^(1/4)';

% Mixture PF parameters
% .....................
% Factors to adjust distance function for k-means [l_m, l_n, l_r]
Params.state_distance_coefficients = [30, 1, 100];
% If distance < cluster_merging_thr: merge clusters
Params.cluster_merging_thr = 20; 
% If spread > cluster_splitting_thr: split clusters
Params.cluster_splitting_thr = 30; 
% If number of clusters > n_max_clusters, kill cluster with lowest weight
Params.n_max_clusters = 100;
% Number of cluster to start with
Params.n_initial_clusters = 32;

% Observation model
% -----------------

% Distribution type {invGauss, fixed, gamma, histogram, multivariateHistogram,
% bivariateGauss, mixOfGauss, MOG, MOG3, ...}
Params.observationModelType = 'MOG';
% Features (extension) to be used
 Params.feat_type{1} = 'lo230_superflux.mvavg';
 Params.feat_type{2} = 'hi250_superflux.mvavg';

% Feature dimension
Params.featureDim = length(Params.feat_type);
% Stores the pattern cluster figures after kmeans if 1
Params.fig_store_flag = 1;
% DATA:
% =====

% Train data
% ----------

% Test data
% ----------

% Params.testLab = 'examples\ex3\testList.lab';

end
