function [k,v]=findPeaks(x,varargin)
%FINDPEAKS finds peaks with optional quadratic interpolation [K,V]= findPeaks(X,varargin)
%
%  Inputs:  X           is the input signal (does not work with UInt datatype)
%           M           is mode:
%                           'q' performs quadratic interpolation
%                           'v' finds valleys instead of peaks
%                           [] does not do anything (default)
%           wdTol       is the width tolerance; a peak will be eliminated if there is
%                       a higher peak within +-W samples (default = 0)
%           ampTol      amplitude tolerance; a peak will be eliminated if it
%                       is less than ampTol (The user needs to take care of the scaling here)
%                       (default = 0.05)            
%           prominence  prominence threshold; a peak will be eliminated if
%                       it is not atleast prominence*valleys sorrounding
%                       the peak. (See topographic prominence of explanation)
%                       (default = 3)
% All inputs except X must be sent using name value pairs
% Outputs:  K        are the peak locations in X (fractional if M='q')
%           V        are the peak amplitudes: if M='q' the amplitudes will be interpolated
%                    whereas if M~='q' then V=X(K). 
% Outputs are column vectors regardless of whether X is row or column.
% If there is a plateau rather than a sharp peak, the routine will place the
% peak in the centre of the plateau. When the W input argument is specified,
% the routine will eliminate the lower of any pair of peaks whose separation
% is <=W; if the peaks have exactly the same height, the second one will be eliminated.
% All peak locations satisfy 1<K<length(X).
%
% If no output arguments are specified, the results will be plotted.
% Example usage:
% [K,V]= findpeaks_new(X,'M','q','wdTol',5,'ampTol',0.1,'prominence',2)
%
%	   Copyright (C) Mike Brookes 2005
%      Version: $Id: findpeaks.m 713 2011-10-16 14:45:43Z dmb $
%
%   VOICEBOX is a MATLAB toolbox for speech processing.
%   Home page: http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   This program is free software; you can redistribute it and/or modify
%   it under the terms of the GNU General Public License as published by
%   the Free Software Foundation; either version 2 of the License, or
%   (at your option) any later version.
%
%   This program is distributed in the hope that it will be useful,
%   but WITHOUT ANY WARRANTY; without even the implied warranty of
%   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%   GNU General Public License for more details.
%
%   You can obtain a copy of the GNU General Public License from
%   http://www.gnu.org/copyleft/gpl.html or by writing to
%   Free Software Foundation, Inc.,675 Mass Ave, Cambridge, MA 02139, USA.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Modified by Ajay Srinivasamurthy, MTG, UPF on 02/08/2013

if nargin < 1
    error('Needs atleast one argument');
end
[M,wdTol,ampTol,prominence] = parse_options_inputs(varargin,'M',[],...
    'wdTol', 0,'ampTol', 0.05, 'prominence', 3);
nx=length(x);
if any(M=='v')
    x=-x(:);        % invert x if searching for valleys
else
    x=x(:);        % force to be a column vector
end
dx=x(2:end)-x(1:end-1);
r=find(dx>0);
f=find(dx<0);

if ~isempty(r) && ~isempty(f)    % we must have at least one rise and one fall
    dr=r;
    dr(2:end)=r(2:end)-r(1:end-1);
    rc=repmat(1,nx,1);
    rc(r+1)=1-dr;
    rc(1)=0;
    rs=cumsum(rc); % = time since the last rise
    
    df=f;
    df(2:end)=f(2:end)-f(1:end-1);
    fc=repmat(1,nx,1);
    fc(f+1)=1-df;
    fc(1)=0;
    fs=cumsum(fc); % = time since the last fall
    
    rp=repmat(-1,nx,1);
    rp([1; r+1])=[dr-1; nx-r(end)-1];
    rq=cumsum(rp);  % = time to the next rise
    
    fp=repmat(-1,nx,1);
    fp([1; f+1])=[df-1; nx-f(end)-1];
    fq=cumsum(fp); % = time to the next fall
    
    k=find((rs<fs) & (fq<rq) & (floor((fq-rs)/2)==0));   % the final term centres peaks within a plateau
    v=x(k);
    
    % now purge nearby peaks
    if wdTol > 0
        j=find(k(2:end)-k(1:end-1)<=wdTol);
        while any(j)
            j=j+(v(j)>=v(j+1));
            k(j)=[];
            v(j)=[];
            j=find(k(2:end)-k(1:end-1)<=wdTol);
        end
    end
    % Now also purge peaks of small prominence and small amplitude
    chosenIndices = zeros(1,length(v));
    diffx = diff([x; x(end)]);
    for p = 1:length(v)
       % Find peak prominence and remove non prominent peaks
       pIndex = k(p);
       lLim = find(diffx(1:pIndex-1) < 0);
       uLim = find(diffx(pIndex+1:end) > 0);
       if isempty(uLim) || isempty(lLim)
           chosenIndices(p) = 1;
       else
           lLim = min(lLim(end)+1,length(x));
           uLim = min(uLim(1)+pIndex,length(x));
           if (abs((v(p)/x(lLim)) > prominence) || abs((v(p)/x(uLim)) > prominence))
                chosenIndices(p) = 1;
           end
       end
       % But then, if it is a prominent high value peak, please keep it
       if (abs(v(p)) > 0.3*max(x))       % Hack to include the big non prominent peaks in
           chosenIndices(p) = 1;
       end
       % Do an amplitude threshold and remove tiny little peaks
       if (abs(v(p)) < ampTol)
           chosenIndices(p) = 0;
       end
       
    end
    v = v(chosenIndices > 0);
    k = k(chosenIndices > 0);
    % do quadratic interpolation
    if any(M=='q')         % do quadratic interpolation
        b=0.5*(x(k+1)-x(k-1));
        a=x(k)-b-x(k-1);
        j=(a>0);            % j=0 on a plateau
        v(j)=x(k(j))+0.25*b(j).^2./a(j);
        k(j)=k(j)+0.5*b(j)./a(j);
        k(~j)=k(~j)+(fq(k(~j))-rs(k(~j)))/2;    % add 0.5 to k if plateau has an even width
    end
    
else
    k=[];
    v=[];
end
if any(M=='v')
    v=-v;    % invert peaks if searching for valleys
end
if ~nargout
    if any(M=='v')
        x=-x;    % re-invert x if searching for valleys
        ch='v';
    else
        ch='^';
    end
    plot(1:nx,x,'-',k,v,ch);
end
