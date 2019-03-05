addpath('/Users/kris/Documents/jayaraman/code/2p')
addpath('/Users/kris/Documents/jayaraman/code/2p/MATLAB-Line-Specific-Code')
addpath('/Users/kris/Documents/jayaraman/code/2p/MATLAB-Analysis-Code')


dir = '/Users/kris/Documents/jayaraman/summer2018/Kris/Documents/imaging/Data_Dan/EB/PEN1_R_EPG_G_EB/'
from_file = load(strcat(dir, 'cont'), 'alldata');

alldata=from_file.alldata;

fly = alldata{1}.allFlyData{4}; %look at fly 4
dat = fly.Dark{3}; %look at trial 3

vR = dat.positionDatMatch.vRot( dat.positionDatMatch.Closed(1:length(dat.positionDatMatch.vRot)) == 1 ) ; %rotational velocity
vF = dat.positionDatMatch.vF(dat.positionDatMatch.Closed(1:length(dat.positionDatMatch.vF))== 1); %forward velocity

ts = dat.positionDatMatch.OffsetRotMatch(:,1);
ts = ts(dat.positionDatMatch.Closed== 1);

L = length(dat.GROIaveMax);

heading = dat.positionDatMatch.PosRotMatch(1:L) * pi/180; %work in radians
heading = heading(dat.positionDatMatch.Closed== 1);

datG = dat.GROIaveMax-1;
datR = dat.RROIaveMax-1;

s = size(datG);
s = s(1);

smooth = 3
%should change to SG smoothing
if smooth > 0
    display('smoothening data');
    L = L-smooth+1;
    ts = Smooth(ts, smooth);
    vR = Smooth(vR, smooth);
    heading = Smooth(heading, smooth);
    vF = Smooth(vF, smooth);
    newG = [];
    newR = [];
    for i = 1:s
        g = datG(i,:);
        newG(i,:) = Smooth(g(dat.positionDatMatch.Closed== 1), smooth);
        r = datR(i,:);
        newR(i,:) = Smooth(r(dat.positionDatMatch.Closed== 1), smooth);
    end
    datG = newG;
    datR = newR;    
end

mG = zeros(L, 1); %magnitude of vector sum of green EB activity
mR = zeros(L, 1);

dG = zeros(L, 1); %direction of vector sum
dR = zeros(L, 1);

for i = 1:L
    %get vector sum and direction for red and green channel
    
    [dGi, mGi] = getVecSum( datG(:, i) );
    dG(i) = dGi;
    mG(i) = mGi;
    
    [dRi, mRi] = getVecSum( datR(:, i) );
    dR(i) = dRi;
    mR(i) = mRi; %note we use magnitude of vector sum, not normalized!
    
end
dR = dR-pi; %work from -pi to pi
dG = dG-pi;

L = length(vR); %number of data points we have velocity for

%%prune away stuff with too low intensity in either channel
prune = 0.35 %threshold is mean times prune

if prune > 0
    rlim = prune*mean(mR);
    glim = prune*mean(mG);   
    dGp = dG( mR > rlim );
    tp = ts( mR > rlim );
    vRp = vR( mR(1:end-1) > rlim ); 
    mGp = mG( mR > rlim );
    dRp = dR( mR > rlim );
    headp = heading( mR > rlim );
    mRp = mR( mR > rlim );
    
    dGp = dGp( mGp > glim );
    tp = tp( mGp > glim );
    vRp = vRp( mGp(1:end-1) > glim ); 
    dRp = dRp( mGp > glim );
    mRp = mRp( mGp > glim );
    headp = headp( mGp > glim );
    mGp = mGp( mGp > glim );
end


rG = getRates(dG, ts, 2*pi); %rate of change of direction of Green vector sum
rR = getRates(dR, ts, 2*pi); %rate of change of direction of Red vector sum
rGp = getRates(dGp, tp, 2*pi);
rRp = getRates(dRp, tp, 2*pi);

%cut off last value to make compatible with our rate data
dG = dG(1:L);
dR = dR(1:L);
mG = mG(1:L);
mR = mR(1:L);
ts = ts(1:L);
heading = heading(1:L);

L2 = length(vRp)
csvwrite('example_data/heading.csv', headp(1:L2))
csvwrite('example_data/direction_EPG.csv', dGp(1:L2))
csvwrite('example_data/direction_PEN1.csv', dRp(1:L2))
csvwrite('example_data/times.csv', tp(1:L2))
csvwrite('example_data/vrot.csv', vRp(1:L2))

csvwrite('example_data/data.csv', [tp(1:L2), ...
    vRp(1:L2), headp(1:L2), dGp(1:L2), dRp(1:L2)])

