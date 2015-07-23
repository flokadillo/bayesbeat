function z = normalizeLogProb(x)
% Normalizes each row of a matrix such that they sum to 1
L = logsumexp(x, 2);
z = bsxfun(@minus, x, L);
% 
% for k = 1:size(y,1)
%     z(k,:) = y(k,:)/(sum(y(k,:)));
% end
