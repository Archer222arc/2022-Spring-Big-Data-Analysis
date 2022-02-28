function py = l1_proj(y,t)
% find the projection of y to l_1 ball with bisection
% Input:
%       y --- the initial point
%       t --- the radius
% Output:
%       py--- the projection
% judge if in l1 box

if norm(y,1) <= t
    py = y;
    return
end

y_sort      = sort(abs(nonzeros(y)),'descend');
y_sum     = cumsum(y_sort);
ndx    = find( y_sum - (1:numel(y_sort))' .* [ y_sort(2:end) ; 0 ] >= t+2*eps(t), 1 ); % For stability
if ~isempty( ndx )
    thresh = ( y_sum(ndx) - t) / ndx;
    py  = y .* ( 1 - thresh ./ max( abs(y), thresh)); % May divide very small numbers
end
