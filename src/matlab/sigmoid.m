function [op, g] = sigmoid(ip)

op = 1 ./ (1 + exp(-ip));
g = op .* (1-op);