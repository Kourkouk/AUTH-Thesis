function x = helperNormalize(x)
% This function is only intended to support this example. It may be changed
% or removed in a future release. 
x = x-median(x);
x = {x/max(x)};
end