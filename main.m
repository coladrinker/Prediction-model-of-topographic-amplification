%%  清空环境变量
warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc                     % 清空命令行

%%  导入数据
res = readmatrix('Zhang数据集.xlsx');
bou = readmatrix('Bouckovalas数据集.xlsx')

%%  划分训练集和测试集
temp = randperm(182);
% temp = 1:1:182;
P_train = res(temp(1: 145), 1: 5)';
T_train = res(temp(1: 145), 6)';
M = size(P_train, 2);

P_test = res(temp(146: end), 1: 5)';
T_test = res(temp(146: end), 6)';
N = size(P_test, 2);

P_bou = bou(:, 1: 5)';
T_bou = bou(:, 6)';
B = size(P_bou, 2);

%%  数据归一化
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);
p_bou = mapminmax('apply', P_bou, ps_input);

[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);
t_bou = mapminmax('apply', T_bou, ps_output);

%%  创建网络
% 隐藏层神经元个数5，若有多个隐藏层则为
% net = newff(p_train, t_train, [5,5]);
net = newff(p_train, t_train, 5);

%%  设置训练参数
net.trainParam.epochs = 1000;     % 迭代次数 
net.trainParam.goal = 1e-6;       % 误差阈值
net.trainParam.lr = 0.01;         % 学习率

%%  训练网络
net = train(net, p_train, t_train);
% 训练后从net中可以查阅权重和偏置等信息
% 权重矩阵  net.IW{1}

%%  仿真测试
t_sim1 = sim(net, p_train);
t_sim2 = sim(net, p_test);
t_sim3 = sim(net, p_bou);

%%  数据反归一化
T_sim1 = mapminmax('reverse', t_sim1, ps_output);
T_sim2 = mapminmax('reverse', t_sim2, ps_output);
T_sim3 = mapminmax('reverse', t_sim3, ps_output);

%%  均方根误差
error1 = sqrt(sum((T_sim1 - T_train).^2) ./ M);
error2 = sqrt(sum((T_sim2 - T_test ).^2) ./ N);
error3 = sqrt(sum((T_sim3 - T_bou ).^2) ./ B);

%%  绘图
% RMSE即为均方根误差
figure
plot(1: M, T_train, 'r-*', 1: M, T_sim1, 'b-o', 'LineWidth', 1)
legend('真实值','预测值')
xlabel('预测样本')
ylabel('预测结果')
string = {'训练集预测结果对比'; ['RMSE=' num2str(error1)]};
title(string)
xlim([1, M])
grid

figure
plot(1: N, T_test, 'r-*', 1: N, T_sim2, 'b-o', 'LineWidth', 1)
legend('真实值','预测值')
xlabel('预测样本')
ylabel('预测结果')
string = {'测试集预测结果对比';['RMSE=' num2str(error2)]};
title(string)
xlim([1, N])
grid

figure
plot(1: B, T_bou, 'r-*', 1: B, T_sim3, 'b-o', 'LineWidth', 1)
legend('真实值','预测值')
xlabel('预测样本')
ylabel('预测结果')
string = {'Bouck集预测结果对比';['RMSE=' num2str(error3)]};
title(string)
xlim([1, B])
grid

% 结果不同的原因
% 1.数据经过了随机打乱
% 2.BP神经网络的初始权重是随机的，可能会对结果构成影响
%%  相关指标计算
% 决定系数 R2
R1 = 1 - norm(T_train - T_sim1)^2 / norm(T_train - mean(T_train))^2;
R2 = 1 - norm(T_test -  T_sim2)^2 / norm(T_test -  mean(T_test ))^2;
R3 = 1 - norm(T_bou -  T_sim3)^2 / norm(T_bou -  mean(T_bou ))^2;

disp(['训练集数据的R2为：', num2str(R1)])
disp(['测试集数据的R2为：', num2str(R2)])
disp(['Bouck集数据的R2为：', num2str(R3)])

% 平均绝对误差 MAE
mae1 = sum(abs(T_sim1 - T_train)) ./ M ;
mae2 = sum(abs(T_sim2 - T_test )) ./ N ;
mae3 = sum(abs(T_sim3 - T_bou )) ./ B ;

disp(['训练集数据的MAE为：', num2str(mae1)])
disp(['测试集数据的MAE为：', num2str(mae2)])
disp(['Bouck集数据的MAE为：', num2str(mae3)])

% 平均相对（应为偏差）误差 MBE
mbe1 = sum(T_sim1 - T_train) ./ M ;
mbe2 = sum(T_sim2 - T_test ) ./ N ;
mbe3 = sum(T_sim3 - T_bou ) ./ B ;

disp(['训练集数据的MBE为：', num2str(mbe1)])
disp(['测试集数据的MBE为：', num2str(mbe2)])
disp(['Bouck集数据的MBE为：', num2str(mbe3)])