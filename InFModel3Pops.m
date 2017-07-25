%% Model a ring attractor with three populations (E-PGs, R P-ENS, L P-ENs)

clear;
close all;
clc;
cd('D:\CompassModel');

%% Set model parameters
% number of neurons per neuron type
nEPG = 54;
nPEN = 9;
EperP = nEPG/nPEN/2;

% time constants
tau(1) = 0.08; % E-PG neural time constant
tau(2) = 0.065; % P-EN neural time constant
tauSyn = 0.01; % synaptic time constant

% resting and threshold potentials
EL = -0.065; % Resting potential
VTh = -0.05; % Threshold potential

% input resistance
RIn = 0.1;

%% Define the weight matrices

% Specify angles for each glomerulus
glomAngs = linspace(-pi,pi,nEPG+1);
glomAngs(end) = [];

% Define a weight matrix for each set of connections
allWs = zeros(nEPG+2*nPEN,nEPG+2*nPEN);

% Create the E-PG to P-EN weights
alpha = 25; % weight multiplier for the direct E-PG to P-EN connections (and vice versa)
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

% Plot the weight matrix
figure('units','normalized','outerposition',[0 0.5 0.5 0.5]);
imagesc(allWs);
caxis([min(min(allWs)) max(max(allWs))]);
axis square;
colorbar;
colormap('jet');

%% Define the time span, the intial conditions, and the velocity

tStep = 1/10000; % time step for the simulation
tSpan = linspace(0, 10, 10/tStep+1); % set the time span of the ring attractor
VAll = zeros(nEPG+2*nPEN,length(tSpan)); % cell voltages
SAll = VAll; % cell spikes
synput = VAll; % synaptic input
IAll= VAll;
VAll(:,1) = zeros(nEPG+2*nPEN,1)+EL; % define the initial conditions
VAll(1:10,1) = -0.055;
vIn = 1; % set a rotational velocity input

%% Solve the ODEs and plot the results

% Precalculate some things
synDecay = exp(-tStep/tauSyn);

% Define the function for the ODE
for tPt = 2:length(tSpan)
    for nron = 1:nEPG+2*nPEN
        % Reset the voltage if it's over the theshold
        if VAll(nron,tPt-1) > VTh
            VAll(nron,tPt) = EL;
            SAll(nron,tPt) = 1;
            synput(nron,tPt) = 1;
        else % Sum the currents using the weight matrices
            synput(nron,tPt) = synput(nron,tPt-1)*synDecay;
            ISum = 0;
            for prtnr = 1:nEPG+2*nPEN
                ISum = ISum + allWs(prtnr,nron)*synput(prtnr,tPt-1);
            end
            IAll(nron,tPt) = ISum;

            % Define the diff. eqs.
            if nron <= nEPG
                VAll(nron,tPt) = VAll(nron,tPt-1) + ...
                    tStep*(1/tau(1)*(-VAll(nron,tPt-1)+EL + RIn*max(ISum,0)));
            elseif nron  <= nEPG+nPEN && vIn >= 0
                VAll(nron,tPt) = VAll(nron,tPt-1) + ...
                    tStep*(1/tau(2)*(-VAll(nron,tPt-1)+EL + RIn*max(ISum+1+vIn,0)));
            elseif nron  <= nEPG+nPEN && vIn < 0
                VAll(nron,tPt) = VAll(nron,tPt-1) + ...
                    tStep*(1/tau(2)*(-VAll(nron,tPt-1)+EL + RIn*max(ISum+1,0)));
            elseif nron  <= nEPG+2*nPEN && vIn > 0
                VAll(nron,tPt) = VAll(nron,tPt-1) + ...
                    tStep*(1/tau(2)*(-VAll(nron,tPt-1)+EL + RIn*max(ISum+1,0)));
            elseif nron  <= nEPG+2*nPEN && vIn <= 0
                VAll(nron,tPt) = VAll(nron,tPt-1) + ...
                    tStep*(1/tau(2)*(-VAll(nron,tPt-1)+EL + RIn*max(ISum+1-vIn,0)));
            end
        end
    end
end

% Plot some things
figure('units','normalized','outerposition',[0.5 0.5 0.5 0.5]);
imagesc(tSpan,[1:nEPG+2*nPEN],VAll);
% caxis([0 1.5*max(max(NVals))]);

figure('units','normalized','outerposition',[0.5 0 0.5 0.5]);
imagesc(tSpan,[1:nEPG+2*nPEN],SAll);