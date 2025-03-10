% Statistical Parameter Estimation, kalman filter
close all
clc
clearvars

% Teht. 2
t = 0:70;
true_states = zeros(size(t));
measurements = zeros(size(t));
state_estimates = zeros(size(t));
state_estimates(1) = 0; % initial state estimate
error_cov = zeros(size(t));
error_cov(1) = 1; % initial error cov

% noise parameters:
mq = 0; mr = 0; Q = 5; R = 2;

for i = 2:length(t)
    % new true state
    true_states(i) = true_states(i-1) + randn()*sqrt(Q) + mq;
    % new measurement
    measurements(i) = true_states(i) + randn()*sqrt(R) + mr;
    % prediction
    pred = state_estimates(i-1);
    pred_cov = error_cov(i-1) + Q;
    error = measurements(i) - pred;
    Sk = pred_cov + R;

    % Kalman gain
    K = pred_cov / Sk;

    % new state estimate
    state_estimates(i) = pred + K*error;

    error_cov(i) = pred_cov - K*Sk*K;
    
end

figure()
plot(t, measurements, "b.")
hold on
plot(t,state_estimates, "k")
legend("Measurements", "state estimates")
title("Kalman filter for Gaussian random walk")
xlabel("t")



%% Teht. 3

k = 5; m = 10; c = 3; % system parameters
A = [0 1; -k/m -c/m];
B = [0 0; 0 1/m];
I = eye(2);
F0 = 1; w = 0.5; % parameters for F(t)
F = @(t) F0.*sin(w.*t);
t = linspace(0,30,100); h = t(2) - t(1);

% initialization
true_states = zeros(2, length(t));
measurements = zeros(2, length(t));
state_estimates = zeros(2, length(t));
state_estimates(:,1) = [0; 0]; % initial state estimate
error_cov = ones(2); % initial error cov

% noise parameters:
mq = [0; 0]; mr = [0; 0]; Q = eye(2); R = 2*eye(2);

for i = 2:length(t)
    % new true state
    true_states(:,i) = (I+A*h)*true_states(:,i-1) + h*B*[1; F(t(i))]...
        + h*chol(Q)*randn(2,1) + mq;
    % new measurement
    measurements(:,i) = true_states(:,i) + chol(R)*randn(2,1) + mr;
    % prediction
    pred = (I+A*h)*state_estimates(:,i-1) + h*B*[1; F(t(i))];
    pred_cov = (I+A*h)*error_cov*(I+A*h)' + Q;
    error = measurements(:,i) - pred;
    Sk = pred_cov + R;
    % Kalman gain:
    K = pred_cov / Sk;
    state_estimates(:,i) = pred + K*error;
    error_cov = pred_cov - K*Sk*K';
end

figure()
plot(t, measurements(1,:), ".b")
hold on
plot(t, state_estimates(1,:), "k")
legend("measurements", "estimates")
title("Kalman filter for forced spring-mass system")

% goodness of fit, r2
SSres = sum( (true_states(1,:) - state_estimates(1,:)).^2 );
SStot = sum( (true_states(1,:) - mean(true_states(1,:))*ones(size(t))).^2 );
R2 = 1 - SSres / SStot



%% Teht. 4

t = linspace(0,5,300); h = t(2) - t(1);
g = 9.81; % putoamiskiihtyvyys

F = @(x) [x(1) + x(2).*h; x(2) - g.*sin(x(1)).*h];
G = @(x) sin(x(1));

JF = @(x) [1 h; -g.*h.*cos(x(1)) 1]; % Jacobian of F
JG = @(x) [cos(x(1)) 0]; % Jacobian of G

% noise parameters:
qc = 0.01; mq = [0; 0]; mr = [0; 0]; 
Q = [qc*h^3/3 qc*h^2/2; qc*h^2/2 qc*h]; R = qc*4;

% initialization
true_states = zeros(2, length(t));
true_states(:,1) = [pi/4; 0];
measurements = zeros(1, length(t));
state_estimates = zeros(2, length(t));
state_estimates(:,1) = [2*pi/4; 0]; % initial state estimate
error_cov = ones(2); % initial error cov

for i = 2:length(t)
    % new true state
    true_states(:,i) = F(true_states(:,i-1))...
        + chol(Q)*randn(2,1) + mq;
    % new measurement
    measurements(i) = G(true_states(1,i))...
        + sqrt(R)*randn();
    % prediction
    pred = F(state_estimates(:,i-1));
    pred_cov = JF(state_estimates(:,i-1))*error_cov*JF(state_estimates(:,i-1))'...
        + Q;
    error = measurements(1,i) - G(pred(1));
    Sk = JG(pred)*pred_cov*JG(pred)' + R;
    % Kalman gain
    K = pred_cov*JG(pred)' / Sk;
    state_estimates(:,i) = pred + K*error;
    error_cov = pred_cov - K*Sk*K';
end

figure()
plot(t,true_states(1,:), "--k", LineWidth=2)
hold on
plot(t, state_estimates(1,:), "b", LineWidth=1.5)
plot(t, measurements, ".r", LineWidth=2)
legend("true angle", "estimated angle", "measurements")
title("Kalman filter for pendulum model")





