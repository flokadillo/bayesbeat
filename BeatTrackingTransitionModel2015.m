classdef BeatTrackingTransitionModel2015 < handle & BeatTrackingTransitionModel
    %UNTITLED4 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        transition_lambda
        tempi_from_pattern
    end
    
    methods
        function obj = BeatTrackingTransitionModel2015(state_space, ...
                transition_params)
            % call superclass constructor
            obj@BeatTrackingTransitionModel(state_space, ...
                transition_params);
            obj.transition_lambda = ...
                transition_params.transition_lambda;
            obj.compute_transitions;
        end
        
    end
    
    methods (Access = protected)
        function [] = compute_transitions(obj)
            % This function creates a transition function with the following
            % properties: each tempostate has a different number of position
            % states
            % only allow transition with probability above tempo_transition_threshold
            tempo_transition_threshold = eps;
            % store variables locally to avoid long filenames
            n_patterns = obj.state_space.n_patterns;
            tempi_from_pattern = obj.state_space.tempi_from_pattern;
            n_position_states = obj.state_space.n_position_states;
            n_beats_from_pattern = obj.state_space.n_beats_from_pattern;
            n_states = obj.state_space.n_states;
            % tempo changes can only occur at the beginning of a beat
            % compute transition matrix for the tempo changes
            tempo_trans = cell(n_patterns, n_patterns);
            % iterate over all tempo states
            for ri = 1:n_patterns
                for rj = 1:n_patterns
                    tempo_trans{ri, rj} = zeros(tempi_from_pattern(ri), ...
                        tempi_from_pattern(rj));
                    for tempo_state_i = 1:length(n_position_states{ri})
                        for tempo_state_j = 1:length(n_position_states{rj})
                            % compute the ratio of the number of beat states 
                            % between the two tempi
                            ratio = n_position_states{rj}(tempo_state_j) /...
                                n_position_states{ri}(tempo_state_i);
                            % compute the probability for the tempo change
                            prob = exp(-obj.transition_lambda * abs(ratio - 1));
                            % keep only transition probabilities > 
                            % tempo_transition_threshold
                            if prob > tempo_transition_threshold
                                % save the probability and apply the
                                % pattern change probability
                                tempo_trans{ri, rj}(tempo_state_i, ...
                                    tempo_state_j) = prob;
                            end
                        end
                        % normalise
                        tempo_trans{ri, rj}(tempo_state_i, :) = ...
                            tempo_trans{ri, rj}(tempo_state_i, :) ./ ...
                            sum(tempo_trans{ri, rj}(tempo_state_i, :));
                    end
                end
            end
            % Apart from the very beginning of a beat, the tempo stays the same,
            % thus the number of transitions is equal to the total number of states
            % plus the number of tempo transitions minus the number of tempo states
            % since these transitions are already included in the tempo transitions
            % Then everything multiplicated with the number of beats with
            % are modelled in the patterns
            n_tempo_transitions = (tempi_from_pattern .* tempi_from_pattern)' * ...
                n_beats_from_pattern(:);
            if obj.state_space.use_silence_state
                obj.n_transitions = n_states + n_tempo_transitions - ...
                    (tempi_from_pattern' * n_beats_from_pattern(:)) + ...
                    2 * sum(tempi_from_pattern) + 1;
            else
                obj.n_transitions = n_states + n_tempo_transitions - ...
                    (tempi_from_pattern' * n_beats_from_pattern(:));
            end
            % initialise vectors to store the transitions in sparse format
            % rows (states at previous time)
            row_i = zeros(obj.n_transitions, 1);
            % cols (states at current time)
            col_j = zeros(obj.n_transitions, 1);
            % transition probabilites
            val = zeros(obj.n_transitions, 1);
            num_states_per_pattern = cellfun(@(x) sum(x), n_position_states) .* ...
                n_beats_from_pattern;
            % get linear index of all beat states
            state_at_beat = cell(n_patterns, max(n_beats_from_pattern));
            si = 0;
            for ri = 1:n_patterns
                % first beat
                state_at_beat{ri, 1} = si + cumsum([1, ...
                    n_position_states{ri}(1:end-1) * ...
                    n_beats_from_pattern(ri)]);
                % subsequent beats
                for bi = 2:n_beats_from_pattern(ri)
                    state_at_beat{ri, bi} = ...
                        state_at_beat{ri, bi-1} + n_position_states{ri};
                end
                si = si + num_states_per_pattern(ri);
            end
            % get linear index of preceeding state of all beat positions
            state_before_beat = cell(n_patterns, max(n_beats_from_pattern));
            for ri = 1:n_patterns
                for bi = 2:n_beats_from_pattern(ri)
                    state_before_beat{ri, bi} = ...
                        state_at_beat{ri, bi} - 1;
                end
                state_before_beat{ri, 1} =  ...
                    state_at_beat{ri, n_beats_from_pattern(ri)} + ...
                    n_position_states{ri} - 1;
            end
            % transition counter
            p = 1;
            for ri = 1:n_patterns
                for ni = 1:tempi_from_pattern(ri)
                    for bi = 1:n_beats_from_pattern(ri)
                        for nj = 1:tempi_from_pattern(ri)
                            if bi == 1 % bar crossing > pattern change?
                                for rj = find(obj.pr(ri, :))
                                    if (tempo_trans{ri, rj}(ni, nj) > 0)
                                        % create transition
                                        % position before beat
                                        row_i(p) = state_before_beat{ri, bi}(ni);
                                        % position at beat
                                        col_j(p) = state_at_beat{rj, bi}(nj);
                                        % store probability
                                        val(p) = tempo_trans{ri, rj}(ni, nj) * ...
                                            obj.pr(ri, rj);
                                        p = p + 1;
                                    end
                                end
                            else
                                if tempo_trans{ri, ri}(ni, nj) ~= 0
                                    % create transition
                                    % position before beat
                                    row_i(p) = state_before_beat{ri, bi}(ni);
                                    % position at beat
                                    col_j(p) = state_at_beat{ri, bi}(nj);
                                    % store probability
                                    val(p) = tempo_trans{ri, ri}(ni, nj);
                                    p = p + 1;
                                end
                            end
                        end % over tempo at current time
                        % transitions of the remainder of the beat: tempo
                        % transitions are not allowed
                        idx = p:p+n_position_states{ri}(ni) - 2;
                        row_i(idx) = state_at_beat{ri, bi}(ni) + ...
                            (0:n_position_states{ri}(ni)-2);
                        col_j(idx) = state_at_beat{ri, bi}(ni) + ...
                            (1:n_position_states{ri}(ni)-1);
                        val(idx) = 1;
                        p = p + length(idx);
                    end % over beats
                end % over tempo at previous time
                if obj.state_space.use_silence_state
                    % transition to silence state possible at bar
                    % transition
                    idx = p:(p + tempi_from_pattern(ri) - 1);
                    row_i(idx) = state_before_beat{ri, 1}(:);
                    col_j(idx) = silence_state_id;
                    val(idx) = obj.p2s;
                    p = p + tempi_from_pattern(ri);
                end
            end % over R
            % add transitions from silence state
            if obj.state_space.use_silence_state
                % one self transition and one transition to each R
                idx = p:(p + sum(tempi_from_pattern));
                % start at silence
                row_i(idx) = silence_state_id;
                % self transition
                col_j(p) = silence_state_id;
                val(p) = 1 - obj.pfs;
                p = p + 1;
                % transition from silence state to m=1, n(:), r(:)
                prob_from_silence = obj.pfs / sum(tempi_from_pattern);
                for i_r=1:n_patterns
                    idx = p:(p + tempi_from_pattern(i_r) - 1);
                    % go to first position of each tempo
                    col_j(idx) = state_at_beat{i_r, 1};
                    val(idx) = prob_from_silence;
                    p = p + tempi_from_pattern(i_r);
                end
            end
            idx = (row_i > 0);
            obj.A = sparse(row_i(idx), col_j(idx), val(idx), ...
                n_states, n_states);
            obj.n_transitions = length(find(obj.A)); 
    end
        
    end
end

