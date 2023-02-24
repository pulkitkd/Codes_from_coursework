%% Generate training data from equations
clc;
clear;

T=10;
dt=0.01;
t=0:dt:T;

b=8/3; 
sig=10; 
r=28;

Lorenz = @(t,x)([ sig * (x(2) - x(1)) ; ...
r * x(1)-x(1) * x(3) - x(2) ; ...
x(1) * x(2) - b*x(3) ]);
ode_options = odeset('RelTol',1e-10, 'AbsTol',1e-11);
input=[]; output=[];
for j=1:100 % training trajectories
    x0=30*(rand(3,1)-0.5);
    [t,y] = ode45(Lorenz,t,x0);
    input=[input; y(1:end-1,:)];
    output=[output; y(2:end,:)];
    plot3(y(:,1),y(:,2),y(:,3)), hold on
    plot3(x0(1),x0(2),x0(3),'ro')
end

%% Train a neural network

net = feedforwardnet([10 10 10]);
net.layers{1}.transferFcn = 'logsig';
net.layers{2}.transferFcn = 'radbas';
net.layers{3}.transferFcn = 'purelin';
net = train(net,input.',output.');

%% Predictions from the neural network

x1 = 30*(rand(3,1)-0.5);
x2 = x1*(1 + 1e-4);

[t,y1] = ode45(Lorenz,t,x1);
plot3(y1(:,1),y1(:,2),y1(:,3),'-')
hold on
plot3(x1(1),x1(2),x1(3),'ro')
hold on
[t,y2] = ode45(Lorenz,t,x2);
plot3(y2(:,1),y2(:,2),y2(:,3),'-')
hold on
plot3(x2(1),x2(2),x2(3),'go')
hold on

% NN prediction
x0 = x1;
ynn(1,:)=x0;
for jj=2:length(t)
    y0=net(x0);
    ynn(jj,:)=y0.';
    x0=y0;
end
plot3(ynn(:,1),ynn(:,2),ynn(:,3),'-.','Linewidth',[1])
hold off

legend('Exact 1','','Exact 2','','NN')

%% Lyapunov time calculation

T=100;
dt=0.01;
t=0:dt:T;

[t,y1] = ode45(Lorenz,t,x1);
[t,y2] = ode45(Lorenz,t,x2);
LE = log(abs((y2(:,1)-y1(:,1))./(x2(1)-x1(1))))./t;
plot(LE(ceil(0.1*length(t)):end))

fprintf("Lyapunov exponent = %f \n",mean(LE(ceil(0.9*length(t)):end)))

