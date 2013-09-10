function [nBars, beatIdx, barStartIdx] = get_full_bars(beats, tolInt, verbose)
%  [nBars, beatIdx] = find_full_bars(beats, [tolInt, verbose])
%  returns complete bars within a sequence of beat numbers
% ----------------------------------------------------------------------
%INPUT parameter:
% beats                     : [nBeats x 2] 
%                               first col: beat times, second col metrical position 
% tolInt                    : pauses are detected if the interval between
%                               two beats is bigger than tolInt times the last beat period
%                               [default 1.8]
%
%OUTPUT parameter:
% nBars                     : number of complete bars
% beatIdx                   : [nBeats x 1] of boolean: determines if beat
%                               belongs to a full bar (=1) or not (=0)
% barStartIdx               : index of first beat of each bar

%
% 11.07.2013 by Florian Krebs
% ----------------------------------------------------------------------
if nargin==1
    tolInt = 1.8;
    verbose = 0;
end

btype = round(rem(beats(:,2),1)*10);
nBeats = length(btype);
meter = max(btype);

% 1) check for pauses
period = diff(beats(:, 1));
ratioPeriod = period(2:end , 1) ./ period(1:end-1 , 1);
btype(find(ratioPeriod>tolInt)+1) = 99;
if verbose, 
   fprintf('%i pauses detected, ', sum(ratioPeriod>tolInt)); 
end
% 2) check for missing or additional beats
array = diff(btype);
if meter == 3
    pattern = [1 1 -2 1];
elseif meter == 4
    pattern = [1 1 1 -3 1];
elseif meter == 8
    pattern = [ones(1, 7), -7, 1];
elseif meter == 9
    pattern = [ones(1, 8), -8, 1];
else
    error('meter %i not known', meter);
end
barStartIdx = strfind(array', pattern);
nBars = length(barStartIdx);
beatIdx = zeros(nBeats, 1);
beatIdx(barStartIdx) = 1;
beatIdx = conv(beatIdx, ones(meter+1, 1));
beatIdx = beatIdx(1:nBeats);
beatIdx(beatIdx~=0) = 1;
if verbose, 
   fprintf('%i beats excluded\n', sum(beatIdx==0)); 
end

end