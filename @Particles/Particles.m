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
            obj.m = zeros(nDiscreteStates, nParticles, nFrames);
            obj.n = zeros(nParticles, nFrames);
            obj.weight = zeros(nParticles, 1);
            obj.posterior_r = zeros(nParticles, nDiscreteStates);
            obj.delta = zeros(nParticles, nDiscreteStates);
            obj.log_trans = zeros(nParticles, nFrames);
            obj.log_obs = zeros(nDiscreteStates, nParticles, nFrames);
            obj.old_weight = zeros(nParticles, 1);
            obj.psi_mat = zeros(nParticles, nDiscreteStates, nFrames, 'uint8');
            obj.nParticles = nParticles;
        end
        
%         function 
%             
%         end
    end
    
end

