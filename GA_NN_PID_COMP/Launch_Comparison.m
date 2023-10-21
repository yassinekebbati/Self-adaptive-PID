%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Yassine Kebbati
% Date: 20/12/2019
% Control NN_PID-GA_PID-COMPARISON
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear
clc

open GAPID_NNPID_COMP.slx;
load NN_weights.mat;

%read the data
data = xlsread('PID_Data.xls');
Kp = data(:,1);
Ki = data(:,2);
Kd = data(:,3);
Theta = data(:,4);
Vwind = data(:,5);
Vref = data(:,6);

%Determine vector dimension base on Vwind data
D1 = length(0:1.15:20); D2 = length(0:1.2:20) ;D3 = length(5:1.4:30);
%length(-15:2.5:15);

%Define learning rate
lr = 0.8;

%transform vertoc data to lookup table
Kp=reshape(Kp,D1,D2,D3);
Ki=reshape(Ki,D1,D2,D3);
Kd=reshape(Kd,D1,D2,D3);

options = simset('SrcWorkspace','current');
S = sim('GAPID_NNPID_COMP',[],options);



