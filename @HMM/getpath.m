function [m, n] = getpath(Meff, annots, frame_length, nFrames)
% [states, m, n] = getpath(Meff, frame_length, annots, duration)
%   Maps the beats to a bar position and tempo for each time instance t
% ----------------------------------------------------------------------
%INPUT parameter:
% Meff              : number of bar positions of model
% annots            : beat annotations (first column = beat times, second column = beat number)
% frame_length      : 
% nFrames           : number of audio frames of song
%
%OUTPUT parameter:
% states            : statenumber at each t: [T x 1]
% m                 : position at each t: [T x 1]
% n                 : tempo at each t: [T x 1]
%
% 3.1.2012 by Florian Krebs
% ----------------------------------------------------------------------
if nargin == 7
    Meff = Meff;
end
btype = annots(:, 2); % position of beat in a bar: 1, 2, 3, 4
if rem(btype(1), 1) > 0
    % old annotation format, where bar id and beat type are stored, e.g.
    % 1.1, 1.2, 1.3, 2.1, 2.1, ...
    btype = round(rem(annots(:, 2), 1) * 10); 
end
if length(btype) == 1 % more than one beat
    fprintf('ERROR getpath: file has only one beat\n');
    n = []; m= []; r = []; states = [];
    return
end
barPointerPosBeats = (Meff * (btype-1)/max(btype))+1;
firstbeats = find(btype==1);
% make the cyclic barPointerPosBeats monotonically increasing by adding
% Meff to each bar
if btype(1)>1
    barPointerPosBeats(firstbeats(1):end) = barPointerPosBeats(firstbeats(1):end) + Meff;
end
for i = 2:length(firstbeats)
    barPointerPosBeats(firstbeats(i):end) = barPointerPosBeats(firstbeats(i):end) + Meff;
end
% skip eventual intros 
t = 0:frame_length:(nFrames-1)*frame_length;
beatTimes = annots(:,1);
% extrapolate beats to before t=0 (for extrapolation)
while beatTimes(1) > 0
    % add additional beat before the sequence
    beatTimes = [2 * beatTimes(1) - beatTimes(2); ...
        beatTimes];
    barPointerPosBeats = [2 * barPointerPosBeats(1) - barPointerPosBeats(2); ...
        barPointerPosBeats];
end
% extrapolate beats to after t=T (for extrapolation)
while beatTimes(end) < (frame_length * nFrames)
    beatTimes = [beatTimes; 2 * beatTimes(end) - beatTimes(end-1)];
    barPointerPosBeats = [barPointerPosBeats; ...
        2 * barPointerPosBeats(end) - barPointerPosBeats(end-1)];
end
barPosPerFrame_linear = interp1(beatTimes, barPointerPosBeats, t, 'cubic', 'extrap');
m = (mod(barPosPerFrame_linear - 1, Meff) + 1)';
n = diff(barPosPerFrame_linear); n = [n, n(end)]';
end

