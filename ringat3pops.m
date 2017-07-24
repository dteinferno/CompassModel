function ringat3pops = odefcn(t,y,nEPG,nPEN,tau,allWs,vIn)

% Total number of neurons
nTot = nEPG+2*nPEN;

% Create the vector of values
ringat3pops = zeros(nEPG+2*nPEN,1);


% Populate the vector
for i = 1:nTot
    % Sum the currents using the weight matrices
    ISum = 0;
    for j = 1:nTot
        ISum = ISum + allWs(j,i)*y(j);
    end
    
    % Define the diff. eqs.
    if i <= nEPG
        ringat3pops(i) = 1/tau(1)*(-y(i) + max(ISum,0));
    elseif i  <= nEPG+nPEN && vIn > 0
        ringat3pops(i) = 1/tau(2)*(-y(i) + max(ISum+1+vIn,0));
    elseif i  <= nEPG+nPEN && vIn < 0
        ringat3pops(i) = 1/tau(2)*(-y(i) + max(ISum+1,0));
    elseif i  <= nEPG+2*nPEN && vIn > 0
        ringat3pops(i) = 1/tau(2)*(-y(i) + max(ISum+1,0));
    elseif i  <= nEPG+2*nPEN && vIn < 0
        ringat3pops(i) = 1/tau(2)*(-y(i) + max(ISum+1-vIn,0));
    end
end