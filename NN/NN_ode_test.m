clc;
clear;

% Define the ODE to be solved
ode = @(y,t) t*sin(t) - sin(y);

% Define the neural network
net = feedforwardnet([10,10,10]);

% Define the loss function
loss = @(y_true, y_pred) mean((y_true - y_pred).^2);

% Generate some training data
t_final = 100;
t_res = 10000;
t_train = linspace(0, t_final, t_res)';
y_train = zeros(size(t_train));
for i = 1:numel(t_train)-1
    y_train(i+1) = y_train(i) + ode(y_train(i), t_train(i))*(t_train(i+1)-t_train(i));
end

% Train the neural network
options = trainingOptions('adam', 'MaxEpochs', 100, 'MiniBatchSize', 10, 'Plots', 'none');
net = train(net, t_train(1:end-1)', y_train(1:end-1)');

%% Evaluate the trained model
t_test = linspace(0, t_final, t_res)';
y_test = zeros(size(t_test));
for i = 1:numel(t_test)-1
    y_test(i+1) = y_test(i) + ode(y_test(i), t_test(i))*(t_test(i+1)-t_test(i));
end
y_pred = net(t_test(1:end-1)');

plot(t_test(1:end-1)', y_test(1:end-1)', 'b-', t_test(1:end-1)', y_pred, 'r--');
legend("True solution", "Predicted solution");
