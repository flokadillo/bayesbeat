classdef Particles < handle
    %This class contains the description of the particles. There are two
    %different cases: 
    %   1) in rao-blackwellized pf, each particle is assigned a vector that
    %   specifies p(r_t | y_1:t, m_1:t, n_1:t)
    %   2) If nDiscreteStates is 1, we sample the whole space
    
    properties
        m                   % position inside a bar state
        n                   % tempo state
        r                   % rhythmic pattern state
        v                   % Section state
        weight
        posterior_r         % p(r_t | z_1:t, y_1:t)
        delta               % prob of the most likely sequence ending in discrete state j at time t, when observing y(1:t)
        psi_mat             % 
        log_trans
        log_obs
        old_weight
        nParticles
        nDiscreteStates
        nFrames
    end
    
    methods
        function obj = Particles(nParticles, nFrames, nDiscreteStates )
            if exist('nDiscreteStates', 'var') && nDiscreteStates > 1 % use rao blackwellized pf
                obj.m = zeros(nDiscreteStates, nParticles, nFrames, 'single');
                obj.posterior_r = zeros(nParticles, nDiscreteStates);
                obj.delta = zeros(nParticles, nDiscreteStates);
                obj.psi_mat = zeros(nParticles, nDiscreteStates, nFrames, 'uint8');
                obj.nDiscreteStates = nDiscreteStates;
            else
                obj.m = zeros(nParticles, nFrames, 'double');
                obj.r = zeros(nParticles, nFrames, 'double');
            end
            obj.n = zeros(nParticles, nFrames, 'double');
            obj.weight = zeros(nParticles, 1);
%             obj.log_trans = zeros(nParticles, nFrames);
%             obj.log_obs = zeros(nDiscreteStates, nParticles, nFrames, 'int8');
            obj.old_weight = zeros(nParticles, 1);
            obj.nParticles = nParticles;
            obj.nFrames = nFrames;
        end
        
        function copyParticles(obj, newIdx)
            if obj.nDiscreteStates > 1 % use rao blackwellized pf
            % copy particles according to newIdx after resampling
                obj.m = obj.m(:, newIdx, :);
                obj.posterior_r = obj.posterior_r(newIdx, :);
                obj.delta = obj.delta(newIdx, :);
                obj.psi_mat = obj.psi_mat(newIdx, :, :);
            else
                obj.m = obj.m(newIdx, :);
                obj.r = obj.r(newIdx, :);
            end
            obj.n = obj.n(newIdx, :);
            obj.weight = log(ones(obj.nParticles, 1) / obj.nParticles);
            
%             obj.log_obs = obj.log_obs(:, newIdx, :);
%             obj.log_trans = obj.log_trans(newIdx, :);
        end
        
        function update_last_particle(obj, newIdx, iFrame)
            if obj.nDiscreteStates > 1 % use rao blackwellized pf
            % copy particles according to newIdx after resampling
                obj.m = obj.m(:, newIdx, :);
                obj.posterior_r = obj.posterior_r(newIdx, :);
                obj.delta = obj.delta(newIdx, :);
                obj.psi_mat = obj.psi_mat(newIdx, :, :);
            else
                obj.m(:, iFrame) = obj.m(newIdx, iFrame);
                obj.r(:, iFrame) = obj.r(newIdx, iFrame);
            end
            obj.n(:, iFrame-1) = obj.n(newIdx, iFrame-1);
            obj.weight = log(ones(obj.nParticles, 1) / obj.nParticles);
            
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

