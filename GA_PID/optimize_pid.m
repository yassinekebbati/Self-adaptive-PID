%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Yassine Kebbati
% Date: 20/12/2019
% Control GA-PID-Autonomous_Driving
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function J = optimize_pid(N,dt,ref,th,vw,parms);

%Set up the different parmeters

set_param('Vehicle_model/Kp','Value','parms(1)');
set_param('Vehicle_model/Ki','Value','parms(2)');
set_param('Vehicle_model/Kd','Value','parms(3)');
set_param('Vehicle_model/Th','Value','th');
set_param('Vehicle_model/Vw','Value','vw');
set_param('Vehicle_model/ref','Value','ref');

options = simset('SrcWorkspace','current');
S = sim('Vehicle_model',[],options);

t = Data.Time;
e = Data.Data(:,1);
r = Data.Data(:,2);
y = Data.Data(:,3);
u = Data.Data(:,4);


%Define the cost function to be optimized
J = 1/N*sum((r(:)-y(:)).^2);


%%%IAE integral of absolute error
%J = dt*abs(r(:)-y(:)).^2)

%%%ITAE intergal of time multiplied by absolute error
%J = dt*sum(t(:).*abs(r(:)-y(:)))


%%%ITSE integral of time multiplied by squared error
%J = dt*sum(t(i)*(r(:)-y(:)).^2)





 