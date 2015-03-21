function [S, t, f] = STFT(x, winsize, hopsize, fftsize, fs, type, online, plots, norm)
%[S, t, f] = STFT(x, winsize, hopsize, fftsize, fs, type, online, plots)
%STFT: Implementation of the Short Time Fourier Transformation
% ----------------------------------------------------------------------
%INPUT parameter:
% x                 : input signal (audio)
% winsize           : window length (fftsize-winsize is padded with zeros)
% hopsize           : hopsize =(1-overlap)*fftsize
% fftsize           : fftsize [samples] (if fftsize > winsize -> zero padding)
% fs                : Sampling Frequency
% plots             : 1 = show plots
% type              : 0 = simple, add zeros at the beginning and end
%                   : 1 = complex, use smaller windows at beginning and end
% online            : only relevant for type=0: 
%                     1 = time frame corresponds to right part of window
%                     0 = time frame corresponds to center of window  
% norm              : 1 = normalize window with sum(w)/2 [default] 
%                   this makes a sinus have a corresponding spectral bin with magnitude 1 (on both positive and negative frequencies)
%               
%
%OUTPUT parameter:
% S                 : Spectrum
% t                 : vector with timeframes
% f                 : vector with frequency frames
%
% 03.05.2011 by Florian Krebs
% 27.04.2011 added simple version
% 26.09.2012 add normalization
% ----------------------------------------------------------------------
if nargin == 5
   type = 0;
   online = 0;
end
if nargin < 8
   plots = 0; 
end
if nargin < 9
   norm = 1; 
end

if size(x,2) > size(x,1)
    x = x';                     % turn input signal x into a [Nx1] column vector
end
if mod(winsize,2) == 0
    winsize = winsize - 1;
end
w = hann(winsize);
if norm, w = w * 2/sum(w); end
nframes = length(x);

xoff = 0;                       % current offset in input signal x
if mod(winsize,2) == 0          % winsize even - rounding is not the perfect solution
    Mo2 = round((winsize-1)/2);
else                            % winsize odd
    Mo2 = (winsize-1)/2;
end

% ----------------------------------------------------------------------
if type == 0
% ----------------------------------------------------------------------
    % simple version: add zeros at beginning and/or end
    nblocks = floor(nframes/hopsize);
    S = zeros(fftsize, nblocks);    % pre-allocate STFT output array
    if online
        % the fft of the window is assigned to the timeframe corresponding to
        % the right side of the window
        x = [zeros(winsize-hopsize,1); x;];
    else
        % the fft of the window is assigned to the timeframe corresponding to
        % the center of the window
        x = [zeros(round((winsize-hopsize)/2),1); x; zeros(round((winsize-hopsize)/2),1)];
    end
    zp = zeros(fftsize-winsize,1);  % zero padding (to be inserted)
    for m=1:nblocks
        xt = x(xoff+1:xoff+winsize);  % extract frame of input data
%         figure; plot(xt(1:10));
        xtw = w .* xt;                % apply window to current frame
%         figure; plot(xtw(1:10));
        % zero padding and circular shifting to calculate phase at the center of
        % the window and not at the beginning
        xtwz = [xtw(Mo2+1:winsize); zp; xtw(1:Mo2)];
        S(:,m) = fft(xtwz);           % STFT for frame m
        xoff = xoff + hopsize;        % advance in-pointer by hop-size
    end
    % ----------------------------------------------------------------------
elseif type == 1
    % ----------------------------------------------------------------------
    % At the beginning we have the problem that we cannot use the full window, because we use only
    % hopsize frames in the center of the window and so we would discard the rest. One solution is
    % to start with smaller windows and increase the size until the whole
    % winsize is covered:
    nblocks = floor((nframes - winsize) / hopsize +1); % number of windows that fit into the signal x
    S = zeros(fftsize, nblocks);    % pre-allocate STFT output array
    hop_per_win = floor(winsize / hopsize);
    if mod(hop_per_win,2) == 0
        hop_per_win = hop_per_win - 1;
    end
    num_win = (hop_per_win + 1) / 2;
    winsize_small = hopsize;
    for n=1:num_win
        xt = x(xoff+1:xoff+winsize_small);  % extract frame of input data
        ws = hann(winsize_small);
        if norm, ws = ws * 2 / sum(ws); end
        xtw = ws .* xt;
        zp = zeros(fftsize-winsize_small,1);
        xtwz = [xtw(round((winsize_small)/2)+1:winsize_small); zp; xtw(1:round((winsize_small)/2))];
        S(:,n) = fft(xtwz);
        winsize_small = winsize_small + 2* hopsize; % increase winsize
    end
    winsize_small = winsize_small - 2* hopsize;
    xoff = winsize_small - round((winsize + hopsize) / 2);
    zp = zeros(fftsize-winsize,1);  % zero padding (to be inserted)
    for m=n+1:(nframes/hopsize)
        xt = x(xoff+1:xoff+winsize);  % extract frame of input data
        xtw = w .* xt;                % apply window to current frame
        % zero padding and circular shifting to calculate phase at the center of
        % the window and not at the beginning
        xtwz = [xtw(Mo2+1:winsize); zp; xtw(1:Mo2)];
        S(:,m) = fft(xtwz);           % STFT for frame m
        xoff = xoff + hopsize;        % advance in-pointer by hop-size
        if (xoff+winsize) > nframes
            lastcenterframe = xoff - hopsize + Mo2;
            RemainingFrames = nframes - (lastcenterframe + hopsize/2);
            winnum = floor(RemainingFrames / hopsize);
            break;
        end
    end
    % smaller windows at the end
    [K,N] = size(S);
    S2 = [S zeros(K,winnum)];
    if (winnum >= 1)
        centerframe = lastcenterframe;
        for m=1:winnum
            centerframe = centerframe + hopsize;
            winsize = round(2*(nframes-centerframe));
            if mod(winsize,2) == 0  %even
                winsize = winsize - 1;
            end
            Mo2 = (winsize-1)/2;
            zp = zeros(fftsize-winsize,1);
            xt = x(nframes-winsize+1:nframes);
            ws = hann(winsize);
            if norm, ws = ws * 2 / sum(ws); end
            xtw = ws .* xt;
            xtwz = [xtw(Mo2+1:winsize); zp; xtw(1:Mo2)];
            S2(:,N+m) = fft(xtwz);
        end
        
    end
    S = S2;  
end

N = size(S,2);
S = S(1:fftsize/2,:);     % throw away negative frequencies
f = (1:fftsize/2) .* fs/fftsize;  % make frequency axis
t = (0:hopsize:nframes) / fs; % time axis in seconds
t = t(1:N);
% if strcmp(type.freq,'midi')
%     [ S, f ] = spec2midi( S, fs );
% end


% --------------------------------------------------------------------
% Plot Spectrogram
% --------------------------------------------------------------------
% compress power envelopes with log-transformation since the
% perception of loudness is roughly proportional to the sum of log-powers
% at critical bands (Klapuri, 2004)
if plots == 1
    magnitude = abs(S);
    P = 20*log10(magnitude+eps);        % compute energy
    figure(length(findobj('Type','figure')) + 1);
    fp = round(f);
    tp = round(t * 1000) / 1000;
    %         P(P < -100) = 0;    % threshhold -100 dB
    imagesc(P);
    numticks = 10;
    
    if numticks > min(length(fp), length(t)), numticks = min(length(fp), length(t)); end;
    set(gca,'YDir','normal'); % displays lower values of y on the bottom, and higher values on top
    set(gca,'YTick',[1:round(length(fp)/numticks):length(f)]) % puts one tick (Beschriftung) at the y axis every 5 data points
    set(gca,'YTickLabel',fp(1:round(length(fp)/numticks):length(fp))) % beschriftet die ticks nach der Liste gegeben in freqs
    set(gca,'XTick',[1:round(length(tp)/numticks):length(tp)]) % puts one tick (Beschriftung) at the y axis every 5 data points
    set(gca,'XTickLabel',tp(1:round(length(tp)/numticks):length(tp))) % beschriftet die ticks nach der Liste gegeben in freqs
    colormap(jet); colorbar;
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
    title('Power Spectrum [dB]');
end
end
