function [m, n] = getpath(M, annots, frame_length, nFrames)
% [states, m, n] = getpath(M, frame_length, annots, duration)
%   Maps the beats to a bar position and tempo for each time instance t
% ----------------------------------------------------------------------
%INPUT parameter:
% M                 : number of bar positions of model
% Meff
% annots            : beat annotations (first column = beat times, second column = beat number)
% duration          : length of signal [sec]
% Meff              : effective number of bar positions of song
%
%OUTPUT parameter:
% states            : statenumber at each t: [T x 1]
% m                 : position at each t: [T x 1]
% n                 : tempo at each t: [T x 1]
%
% 3.1.2012 by Florian Krebs
% ----------------------------------------------------------------------
if nargin == 7
    Meff = M;
end
% effective number of position states of meter
%     Meff = round(M*meter/4);

btype = round(rem(annots(:,2),1)*10); % position of beat in a bar: 1, 2, 3, 4
if length(btype) == 1 % more than one beat
    fprintf('ERROR getpath: file has only one beat\n');
    n = []; m= []; r = []; states = [];
    return
end
if ismember(max(btype), [3, 4])
    met_denom = 4;
elseif ismember(max(btype), [8, 9])
    met_denom = 8;
else
    error('Meter %i invalid !', max(btype));
end

% barPointerPosBeats = (Meff * (btype-1)/4)+1; % bar-pointer position of beat in a bar: 1, 2, 3, 4
barPointerPosBeats = (M * (btype-1)/met_denom)+1;
firstbeats = find(btype==1);

% make the cyclic barPointerPosBeats monotonically increasing by adding
% M to each bar
if btype(1)>1
    barPointerPosBeats(firstbeats(1):end) = barPointerPosBeats(firstbeats(1):end) + M;
end
for i = 2:length(firstbeats)
    barPointerPosBeats(firstbeats(i):end) = barPointerPosBeats(firstbeats(i):end) + M;
end

% skip eventual intros 
% if annots(1, 1) > diff(annots(1:2, 1)) * 1.2
%     t = annots(1, 1):frame_length:annots(end, 1) + frame_length;
% else
    t = 0:frame_length:(nFrames-1)*frame_length;
% end

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
m = (mod(barPosPerFrame_linear - 1, M) + 1)';
n = diff(barPosPerFrame_linear); n = [n, n(end)]';

% if strcmp(params.inferenceMethod, 'HMM_viterbi')
%     
%     n_HMM = zeros(size(n));
%     % replace artificial first beat by bar position of the first frame
%     barPointerPosBeats(1) = barPosPerFrame_linear(1);
%     barPointerPosBeats(end) = barPosPerFrame_linear(end);
%     beatTimes(1) = t(1);
%     beatTimes(end) = t(end);
%     beatFrames = floor((beatTimes-t(1))/frame_length) + 1;
%  
%     ibi_frames = diff(barPointerPosBeats) ./ round(diff(beatTimes) ./ frame_length);
%     valid = ~isinf(ibi_frames);
%     ibi_frames = ibi_frames(valid);
%     lo_ibi = floor(ibi_frames);
%     hi_ibi = ceil(ibi_frames);
%     b=round(diff(beatTimes) ./ frame_length);
%     b = b(valid);
%     n1 = round(b .* (ibi_frames - hi_ibi) ./ (lo_ibi - hi_ibi));
%     n1(lo_ibi == hi_ibi) = b(lo_ibi == hi_ibi);
%     n2 = b - n1;
%     ibi_frames = [ibi_frames; ibi_frames(end)];
%     if ibi_frames(1) < ibi_frames(2)
%         n_HMM(1:b(1)) = [ones(n1(1), 1) * lo_ibi(1); ones(n2(1), 1) * hi_ibi(1)];
%     else
%         n_HMM(1:b(1)) = [ones(n2(1), 1) * hi_ibi(1); ones(n1(1), 1) * lo_ibi(1)];
%     end
%     
%     for iBeats = 2:length(beatTimes)-1
%         
%         lastn = n_HMM(beatFrames(iBeats)-1);
%         nextn = ibi_frames(iBeats+1);
%         if ibi_frames(iBeats) > nextn, nextn = ceil(nextn);
%         else  nextn = floor(nextn); end
%         nFrames = beatFrames(iBeats+1) - beatFrames(iBeats);
%         nPos = 300;
%         nCurr = lastn;
%         for iF = beatFrames(iBeats):beatFrames(iBeats+1)-1
%             n_HMM(iF) = nCurr;
%             frames2go = beatFrames(iBeats+1) - iF - 1;
%             pos2go = 300 - sum(n_HMM(beatFrames(iBeats):iF));
%             nRatio = pos2go / frames2go;
%             if (nRatio - nCurr) > 1
%                 nCurr = nCurr + 1;
%             elseif (nRatio - nCurr) < -1
%                 nCurr = nCurr - 1;
%             else
%                 if abs(nextn - nCurr) <= 1
%                     lo_ibi = floor(nRatio);
%                     hi_ibi = ceil(nRatio);
%                     n1 = round(frames2go .* (nRatio - hi_ibi) ./ (lo_ibi - hi_ibi));
%                     if lo_ibi == hi_ibi, n1 = frames2go; end
%                     n2 = frames2go - n1;
%                     if nextn >= nCurr
%                         n_HMM(iF+1:iF+n1) = lo_ibi;
%                         n_HMM(iF+n1+1:iF+frames2go) = hi_ibi;
%                         break;
%                     else
%                         n_HMM(iF+1:iF+n2) = hi_ibi;
%                         n_HMM(iF+n2+1:iF+frames2go) = lo_ibi;
%                         break;
%                     end
%                 else
% %                     if abs(nextn - nCurr) <= 1
%                 end
%             end
%         end
%     end
%     n_changes = sum(abs(diff(n_HMM)));
%     m = round(m);
%     
% elseif strcmp(params.inferenceMethod, 'PF')
%     % bar position at each time frame
%     
%     n_changes = []; % only for HMMs
% end
% 
% if sum(n>N) > 0
%     fprintf('WARNING getpath: maximal N exceeded\n');
% end
% n = n';
% n = [n; n(end)]; n(n>N) = N;
% m(m>Meff) = Meff; n(n>N) = N;
% states = sub2ind([M N params.R], round(m), round(n), ones(length(m),1)*tm);
% r = ones(size(m)) * tm;

end

