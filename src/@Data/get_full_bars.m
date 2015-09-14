function [nBars, beatIdx, barStartIdx, predominant_meter] = ...
    get_full_bars(beats, tolInt, verbose)
%  [nBars, beatIdx, barStartIdx] = get_full_bars(beats, tolInt, verbose)
%  returns complete bars within a sequence of beats. If there are multiple
%  time signature within the beat sequence, only the main meter is
%  counted.
% ----------------------------------------------------------------------
%INPUT parameter:
% beats                     : [nBeats x 2]
%                               first col: beat times, second col metrical position
% tolInt                    : pauses are detected if the interval between
%                               two beats is bigger than tolInt times the last beat period
%                               [default 1.8]
%
%OUTPUT parameter:
% nBars                     : number of complete bars [1]
% beatIdx                   : [nBeats x 1] of boolean: determines if beat
%                               belongs to a full bar (=1) or not (=0). The
%                               first beat after a full bar is considered
%                               as belonging to the full bar too
%                               (sum(beatIdx) = nBars * predominant_meter)
% barStartIdx               : index of first beat of each bar
% predominant_meter         : number of beats per bar of the pre-dominant
%                               time signature found in the piece
%
% 11.07.2013 by Florian Krebs
% ----------------------------------------------------------------------
if nargin==1
    tolInt = 1.8;
    verbose = 0;
end
if size(beats, 2) == 3
    btype = beats(:, 3);
else
    btype = beats(:, 2);
end
nBeats = length(btype);
% find most frequent occuring maximum beat type
frequency = histc(btype, 1:max(btype));
predominant_meter = -1;
for i_meter = max(btype):-1:2
    % the main meter should be most frequent meter in the piece
    is_most_frequent = all(frequency(i_meter) >= frequency(2:i_meter-1) - ...
        frequency(i_meter));
    if is_most_frequent
        predominant_meter = i_meter;
        break;
    end
end
if predominant_meter == -1
    fprintf('WARNING: no predominant time signature found')
end
% 1) check for pauses
period = diff(beats(:, 1));
ratioPeriod = period(2:end , 1) ./ period(1:end-1 , 1);
btype(find(ratioPeriod>tolInt)+1) = 99;
if verbose,
    fprintf('%i pauses detected, ', sum(ratioPeriod>tolInt));
end
% 2) check for missing or additional beats
array = diff(btype);
pattern = [ones(1, predominant_meter-1), -(predominant_meter-1), 1];
barStartIdx = strfind(array', pattern);
nBars = length(barStartIdx);
beatIdx = zeros(nBeats, 1);
beatIdx(barStartIdx) = 1;
beatIdx = conv(beatIdx, ones(predominant_meter + 1, 1));
beatIdx = logical(beatIdx(1:nBeats));
if verbose,
    fprintf('%i beats excluded\n', sum(beatIdx==0));
end

end