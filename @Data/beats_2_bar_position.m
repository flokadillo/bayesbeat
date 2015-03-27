function m = beats_2_bar_position(beats, M, frame_length)
btype = round(rem(annots(:,2),1)*10); % position of beat in a bar: 1, 2, 3, 4
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
end