% 五次多项式轨迹规划 (0 到 pi/2，10ms 一个点，0-10s)
clc;
clear;
close all;
%% 机械臂设置
initial_q = [0.0297,0.314,0.0297,-0.775,-0.178,0.356,0]*0;
% 创建q1数组，第一列是0-10的时间点，第二列均为initial_q(1)
time_points = linspace(0, 10, 1000)';  % 创建1000个时间点，从0到10
q1_values = repmat(initial_q(1), 1000, 1);  % 创建1000个相同的initial_q(1)值
q2_values = repmat(initial_q(2), 1000, 1);  % 创建1000个相同的initial_q(2)值
q3_values = repmat(initial_q(3), 1000, 1);  % 创建1000个相同的initial_q(3)值
q4_values = repmat(initial_q(4), 1000, 1);  % 创建1000个相同的initial_q(4)值
q5_values = repmat(initial_q(5), 1000, 1);  % 创建1000个相同的initial_q(5)值
q6_values = repmat(initial_q(6), 1000, 1);  % 创建1000个相同的initial_q(6)值
q7_values = repmat(initial_q(7), 1000, 1);  % 创建1000个相同的initial_q(7)值
Joint_angles.q1 = [time_points, q1_values];  % 组合成两列数组
Joint_angles.q2 = [time_points, q2_values];  % 组合成两列数组
Joint_angles.q3 = [time_points, q3_values];  % 组合成两列数组
Joint_angles.q4 = [time_points, q4_values];  % 组合成两列数组
Joint_angles.q5 = [time_points, q5_values];  % 组合成两列数组
Joint_angles.q6 = [time_points, q6_values];  % 组合成两列数组
Joint_angles.q7 = [time_points, q7_values];  % 组合成两列数组

% 参数设置
t_start = 0;        % 起始时间 (s)
t_end = 10;         % 终止时间 (s)
dt = 0.01;          % 时间间隔 (10ms)
t = t_start:dt:t_end; % 时间向量 (0, 0.01, 0.02, ..., 10)

q_start = pi/2;        % 起始位置 (0)
q_end = 0;       % 终止位置 (π/2)

% 五次多项式系数计算 (q(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5)
% 边界条件：起始和终止时刻的位置、速度、加速度均为0
A = [1, t_start, t_start^2, t_start^3, t_start^4, t_start^5;
     0, 1,        2*t_start, 3*t_start^2, 4*t_start^3, 5*t_start^4;
     0, 0,        2,         6*t_start,   12*t_start^2, 20*t_start^3;
     1, t_end,    t_end^2,   t_end^3,     t_end^4,     t_end^5;
     0, 1,        2*t_end,   3*t_end^2,   4*t_end^3,   5*t_end^4;
     0, 0,        2,         6*t_end,     12*t_end^2,  20*t_end^3];

b = [q_start; 0; 0; q_end; 0; 0]; % 位置、速度、加速度边界条件

a = A \ b; % 求解五次多项式系数 [a0; a1; a2; a3; a4; a5]

% 计算位置、速度、加速度
q = a(1) + a(2)*t + a(3)*t.^2 + a(4)*t.^3 + a(5)*t.^4 + a(6)*t.^5; % 位置
qd = a(2) + 2*a(3)*t + 3*a(4)*t.^2 + 4*a(5)*t.^3 + 5*a(6)*t.^4;    % 速度
qdd = 2*a(3) + 6*a(4)*t + 12*a(5)*t.^2 + 20*a(6)*t.^3;             % 加速度
%% 新增部分：从 -pi/2 到 -pi 的五次多项式轨迹规划（与原代码独立）
% 参数设置
q_start_new = -pi/2;    % 起始位置 (-π/2)
q_end_new = -pi;        % 终止位置 (-π)

% 边界条件矩阵（形式与原有代码一致）
A_new = [1, t_start, t_start^2, t_start^3, t_start^4, t_start^5;
         0, 1,        2*t_start, 3*t_start^2, 4*t_start^3, 5*t_start^4;
         0, 0,        2,         6*t_start,   12*t_start^2, 20*t_start^3;
         1, t_end,    t_end^2,   t_end^3,     t_end^4,     t_end^5;
         0, 1,        2*t_end,   3*t_end^2,   4*t_end^3,   5*t_end^4;
         0, 0,        2,         6*t_end,     12*t_end^2,  20*t_end^3];

b_new = [q_start_new; 0; 0; q_end_new; 0; 0]; % 位置、速度、加速度边界条件

% 求解系数
a_new = A_new \ b_new; % 新的系数 [a0_new; a1_new; ...; a5_new]

% 计算新轨迹的位置、速度、加速度（变量名加后缀 _new 避免冲突）
q_new = a_new(1) + a_new(2)*t + a_new(3)*t.^2 + a_new(4)*t.^3 + a_new(5)*t.^4 + a_new(6)*t.^5;
qd_new = a_new(2) + 2*a_new(3)*t + 3*a_new(4)*t.^2 + 4*a_new(5)*t.^3 + 5*a_new(6)*t.^4;
qdd_new = 2*a_new(3) + 6*a_new(4)*t + 12*a_new(5)*t.^2 + 20*a_new(6)*t.^3;

% 绘制结果
figure;
subplot(3,1,1);
plot(t, q, 'b', 'LineWidth', 1.5);
ylabel('位置 (rad)');
title('五次多项式轨迹规划 (0 到 \pi/2)');
grid on;

subplot(3,1,2);
plot(t, qd, 'r', 'LineWidth', 1.5);
ylabel('速度 (rad/s)');
grid on;

subplot(3,1,3);
plot(t, qdd, 'g', 'LineWidth', 1.5);
ylabel('加速度 (rad/s^2)');
xlabel('时间 (s)');
grid on;

% 生成时间和关节角度的数组
joint_angles = [t', -q']; % 第一列为时间，第二列为关节角度
joint_angles_minus = [t', -q']; % 第一列为时间，第二列为关节速度
joint_angles_45 = [t', -(q)']; % 第一列为时间，第二列为关节角度
joint_angles_45_2 = [t', q_new']; % 第一列为时间，第二列为关节角度
q=-q;
joint_angles_bottom = [t', -q']; % 第一列为时间，第二列为关节角度
joint_angles_minus_bottom = [t', -q']; % 第一列为时间，第二列为关节速度
joint_angles_45_bottom = [t', (2*q)']; % 第一列为时间，第二列为关节角度
joint_angles_45_bottom_2 = [t', -q_new']; % 第一列为时间，第二列为关节角度
% 输出数据（可选）
disp('前5个数据点（时间和关节角度）：');
disp(joint_angles(1:5, :));