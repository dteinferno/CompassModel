%% Model a ring attractor with three populations (E-PGs, R P-ENS, L P-ENs)

%% Set model parameters
% number of neurons per neuron type
nEPG = 54;
nPEN = 9;
EperP = nEPG/nPEN/2;

% time constants
tau(1) = 0.08; % E-PG neural time constant
tau(2) = 0.065; % P-EN neural time constant

%% Define the weight matrices

% Specify angles for each glomerulus
glomAngs = linspace(-pi,pi,nEPG+1);
glomAngs(end) = [];

% Define a weight matrix for each set of connections
allWs = zeros(nEPG+2*nPEN,nEPG+2*nPEN);

% Create the E-PG to P-EN weights
alpha = 10; % weight multiplier for the direct E-PG to P-EN connections (and vice versa)
beta = 25; % weight multiplier for inhibitory  E-PG to P-EN connections

allWs(1:nEPG,nEPG+1:nEPG+2*nPEN) = -beta/nEPG;
allWs(1:EperP,nEPG+1)=allWs(EperP,nEPG+1:nEPG+1)+2*alpha/nEPG;
allWs(EperP+1:2*EperP,nEPG+nPEN+1)=allWs(nPEN*EperP+(1:EperP),nEPG+nPEN+1)+2*alpha/nEPG;
for it = 2:nPEN
    allWs(1:nEPG,nEPG+it) = circshift(allWs(1:nEPG,nEPG+it-1),2*EperP);
    allWs(1:nEPG,nEPG+nPEN+it) = circshift(allWs(1:nEPG,nEPG+nPEN+it-1),2*EperP);
end


% Create the P-EN to E-PG weights
anatShift = 35*pi/180;
kappa = 12;

rXn = alpha*(0.5*vonMises(glomAngs,pi,kappa)+vonMises(glomAngs,pi+anatShift,kappa));
allWs(nEPG+1,1:nEPG) = rXn/nPEN;
lXn = alpha*(0.5*vonMises(glomAngs,pi,kappa)+vonMises(glomAngs,pi-anatShift,kappa));
allWs(nEPG+nPEN+1,1:nEPG) = lXn/nPEN;

for it = 2:nPEN
    allWs(nEPG+it,1:nEPG) = circshift(allWs(nEPG+it-1,1:nEPG),2*EperP,2);
    allWs(nEPG+nPEN+it,1:nEPG) = circshift(allWs(nEPG+nPEN+it-1,1:nEPG),2*EperP,2);
end

% Plot the weights
figure;
imagesc(allWs);
caxis([min(min(allWs)) max(max(allWs))]);
axis square;
colorbar;
colormap('jet');

%% Define the time span, the intial conditions, and the velocity input

tSpan = [0 200]; % set the time span of the ring attractor
y0 = zeros(nEPG+2*nPEN,1); % define the initial conditions
vIn = 0.01;

%% Solve the ODEs and plot the results
% Define the function for the ODE
[tPts NVals] = ode45(@(t,y) ringat3pops(t,y,nEPG,nPEN,tau,allWs,vIn), tSpan, y0);

% Plot some things
figure;
imagesc(tPts,[1:nEPG+2*nPEN],NVals');
caxis([0 1.5*max(max(NVals))]);