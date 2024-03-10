clear; clc; close all;

robot = loadrobot("universalUR5e")
showdetails(robot)

%% 
% Inverse dynamics control.
%%
robot.DataFormat = 'column';
robot.Gravity = [0, 0, -9.82];

%% 
% Simulation
natural_frequency = 2;          % Set value
damping_ratio = 2;              % Set value

kp = natural_frequency^2 * eye(3);
kd = 2 * damping_ratio * natural_frequency * eye(3);

q0 = [0, -2*pi/3, pi/2, -pi/4, -pi/2, 0]';                % Set value

dq0 = [0, 0, 0, 0, 0, 0]';

x_d = [0.5, 0.5, 0.5]';              % Set value
show(robot, q0)

odefun = @(t, y) dydt_IDC(t, y, kp, kd, robot, x_d);

tic
[t, y] = ode45(odefun, [0 5], [q0; dq0]);
toc

%% plots
sim_q = y(:, 1:6);
sim_dq = y(:, 7:12);

framesPerSecond = 60;
r = rateControl(framesPerSecond);
for i = 1:3:size(sim_q, 1)
    T_base_flange = robot.getTransform(sim_q(i, :)', "base", "flange");
    x_e = T_base_flange(1:3, 4)
    show(robot,sim_q(i,:)','PreservePlot',false);
    xlim([-1 1])
    ylim([-1 1])
    zlim([0 1])
    drawnow
    waitfor(r);
end

set(groot,'defaultAxesTickLabelInterpreter','latex');  
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');

fig = figure;
fig.Units               = 'centimeters';
fig.Position(3)         = 16; % width
fig.Position(4)         = 14; % height

subplot(4,1,1:2)

plot(t, sim_q)
hold on

colors = lines(3);

plot(t, interp1([0 5], [q0(1) qref(1)], t), "Color", colors(1, :), 'LineStyle', '--')
plot(t, interp1([0 5], [q0(2) qref(2)], t), "Color", colors(2, :), 'LineStyle', '--')
plot(t, interp1([0 5], [q0(3) qref(3)], t), "Color", colors(3, :), 'LineStyle', '--')

% yline(qref, '--')
ylabel('$q$ [rad]');
grid on
xticklabels({})
yticks([0, pi/3, pi/2, 3*pi/2])
yticklabels({'0', '$\frac{\pi}{3}$\enspace', '$\frac{\pi}{2}$', '$\frac{3 \pi}{2}$'})
legend(["Joint 1", "Joint 2", "Joint 3"], ...
    'NumColumns', 1, ...
    'Location', 'northeast')


subplot(4,1,3:4)
plot(t, sim_dq)
grid on
ylabel('$\dot{q}$ [rad/s]')
xlabel("Time [s]")

xlim([0, t(end)])

exportgraphics(fig,'ex5_IDC.pdf', 'BackgroundColor', 'none') 

function dydt = dydt_IDC(t, y, kp, kd, robot, x_d)
%DQDT_IDC Computes the dydt based on Inverse Dynamic Control and is used for ode45.
%   dydt = dydt_PD_gravity(t, y, kp, kd, robot, q0, qf) returns the derivatives [dq; ddq, dV]. 
% y represents the robot configuration, y = [q, dq, V] at a given time step, t.
% kp and kd represent the proportional and derivative gains respectively.
% robot represents the ur5e_3 rigidBodyTree from previous lectures with the Robotics System Toolbox.
% q0 is the start configuration
% qf is the input argument for the final desired configuration.
disp(t)

q = y(1:6);
dq = y(7:12);

B = robot.massMatrix(q);
Cdq = robot.velocityProduct(q, dq);
grav = robot.gravityTorque(q);

J = geometricJacobian(robot, q, "flange"); % linear velocity part of jacobians are the same, so analytical is not nessesary
Jv = J(1:3, :); % reduce jacobian to linear velocity only

T_base_flange = robot.getTransform(q, "base", "flange");
x_e = T_base_flange(1:3, 4); % pretty sure this is the death sin
x_tilde = x_d - x_e;

%y = ddqref + kd*(dqref-dq) + kp*(qref-q);
%u = B*y + Cdq + grav; % Calculate control effort.

% PD control with gravity compensation for operational space control
u = grav + Jv'*kp*x_tilde - Jv'*kd*Jv*dq;

% Robot dynamics
ddq = inv(B) * (u - Cdq - grav);

dydt = [dq; ddq];

end