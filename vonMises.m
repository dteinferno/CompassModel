function vMVals = vonMises(X, mu, kappa)
% Create a von Mises distribution
% exp(kappa*cos(X-mu))./(2*pi*besseli(0,kappa))
% X = vector of points around the circle (should span 2pi)
% mu = location of center
% kappa = concentration

vMVals = exp(kappa*cos(X-mu))./(2*pi*besseli(0,kappa));

end