%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Yassine Kebbati
% Date: 20/12/2019
% Control GA-PID-Autonomous_Driving
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clear all, close all, clc

open Vehicle_model.slx;
PopSize = 12;
MaxGenerations = 7;
%CrossoverFraction = 0.5;

dt = 1e-3;
N = 30/dt;

%Define lower and upper bounds
lb = [0.1;0.1;0.1];
ub = [1000;1000;1000];
i=0;

% Create loop with changing parameters for optimization
for Vref =5:5:30;
  for Theta =4:1:10;
     for Vwind= -16:3:16;

        i = i+1;
        options = optimoptions(@ga,'PopulationSize',PopSize,'MaxGenerations',MaxGenerations,'OutputFcn',@myfun);%, 'CrossoverFraction',CrossoverFraction);
        [x,fval] = ga(@(K)optimize_pid(N,dt,Vref,Theta,Vwind,K),3,-eye(3),zeros(3,1),[],[],lb ,ub,[],options)

        % %save('x')
        X(i,:) = x; %save data
        % %J(i,1) = fval;
        
        Theta(i,1) = Theta;
        Vwind(i,1) = Vwind;
        R(i,1) = Vref;

        %Save optimization data to excel file
        xlswrite('GA_DATA',{'Kp' 'Ki' 'Kd' 'Theta' 'Vwind' 'Vref'},1,'A1')
        xlswrite('GA_DATA',[X(:,1),X(:,2),X(:,3),Theta,Vwind,R,],1,'A2')

  end
  end
  end