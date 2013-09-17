classdef Particles
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        m
        n
        weight
        posterior_r         % p(r_t | z_1:t, y_1:t)
        delta               % prob of the most likely sequence ending in discrete state j at time t, when observing y(1:t)
        psi_mat             % 
        log_trans
        log_obs
        old_weight
        nParticles
    end
    
    methods
        function obj = Particles(nParticles, nDiscreteStates, nFrames)
            obj.m = zeros(nDiscreteStates, nParticles, nFrames, 'single');
            obj.n = zeros(nParticles, nFrames, 'single');
            obj.weight = zeros(nParticles, 1);
            obj.posterior_r = zeros(nParticles, nDiscreteStates);
            obj.delta = zeros(nParticles, nDiscreteStates);
%             obj.log_trans = zeros(nParticles, nFrames);
%             obj.log_obs = zeros(nDiscreteStates, nParticles, nFrames, 'int8');
            obj.old_weight = zeros(nParticles, 1);
            obj.psi_mat = zeros(nParticles, nDiscreteStates, nFrames, 'uint8');
            obj.nParticles = nParticles;
        end
        
        function obj = copyParticles(obj, newIdx)
            % copy particles according to newIdx after resampling
            obj.m = obj.m(:, newIdx, :);
            obj.n = obj.n(newIdx, :);
            obj.posterior_r = obj.posterior_r(newIdx, :);
            obj.weight = ones(obj.nParticles, 1) / obj.nParticles;
            obj.delta = obj.delta(newIdx, :);
            obj.psi_mat = obj.psi_mat(newIdx, :, :);
%             obj.log_obs = obj.log_obs(:, newIdx, :);
%             obj.log_trans = obj.log_trans(newIdx, :);
        end
        
        function obj = update_m(obj, m_new, iFrame)
            obj.m(:, :, iFrame) = m_new;
        end
%         function 
%             
%         end
    end
    
end

