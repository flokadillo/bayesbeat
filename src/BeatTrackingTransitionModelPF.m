classdef BeatTrackingTransitionModelPF
    % BeatTrackingTransitionModelPF
    % Define the Particle Filter transition model
    
    properties
        state_space
        pr
        pfs
        p2s
        tempo_ss_std_per  % Tempo std in percentage of the absolute tempo
        min_tempo_ss
        max_tempo_ss
    end
    
    methods
        function obj = BeatTrackingTransitionModelPF(state_space, ...
                transition_params)
            obj.state_space = state_space;
            obj.pr = transition_params.pr;
            if obj.state_space.use_silence_state
                obj.pfs = transition_params.pfs;
                obj.p2s = transition_params.p2s;
            else
                obj.pfs = 0;
                obj.p2s = 0;
            end
            obj.tempo_ss_std_per = transition_params.tempo_std_per;
            obj.min_tempo_ss = obj.state_space.convert_tempo_from_bpm(...
                obj.state_space.min_tempo_bpm);
            obj.max_tempo_ss = obj.state_space.convert_tempo_from_bpm(...
                obj.state_space.max_tempo_bpm);
            
        end
        
        function tempo_new = sample_tempo(obj, tempo_old, pattern_new)
            tempo_new = tempo_old + tempo_old .* obj.tempo_ss_std_per ...
                .* randn(length(tempo_old), 1);
            % sampled tempo is clipped to the tempo limits
            out_of_range = tempo_new > obj.max_tempo_ss(pattern_new);
            tempo_new(out_of_range) = ...
                obj.max_tempo_ss(pattern_new(out_of_range));
            out_of_range = tempo_new < obj.min_tempo_ss(pattern_new);
            tempo_new(out_of_range) = ...
                obj.min_tempo_ss(pattern_new(out_of_range));
        end
        
        function position_new = update_position(obj, position_old, tempo_old, ...
                pattern_old)
            position_new = position_old + tempo_old;
            position_new = mod(position_new - 1, ...
                obj.state_space.max_position_from_pattern(pattern_old)) + 1;
        end
        
        function pattern_new = sample_pattern(obj, pattern_old, position_new, ...
                position_old, tempo_old)
            % Pattern transitions to be handled here
            pattern_new = pattern_old;
            % Change the ones for which the bar changed
            crossed_barline = find(position_new < position_old);
            for ri = 1:obj.state_space.n_patterns
                idx_ri = find(pattern_old(crossed_barline) == ri);
                if ~isempty(idx_ri)
                    % sample new pattern from pr
                    pattern_new_s = randsample(obj.state_space.n_patterns, ...
                        length(idx_ri), true, obj.pr(ri, :));
                    % check if the tempo of a particle fits to the tempo range
                    % of the new pattern
                    tempo_valid = (obj.min_tempo_ss(pattern_new_s) <= ...
                        tempo_old(crossed_barline(idx_ri)) <=...
                        obj.max_tempo_ss(pattern_new_s));
                    pattern_new(crossed_barline(idx_ri(tempo_valid))) = ...
                        pattern_new_s(tempo_valid);
                end
            end
        end
    end
    
end

