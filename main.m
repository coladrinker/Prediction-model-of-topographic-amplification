%%  ��ջ�������
warning off             % �رձ�����Ϣ
close all               % �رտ�����ͼ��
clear                   % ��ձ���
clc                     % ���������

%%  ��������
res = readmatrix('Zhang���ݼ�.xlsx');
bou = readmatrix('Bouckovalas���ݼ�.xlsx')

%%  ����ѵ�����Ͳ��Լ�
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

%%  ���ݹ�һ��
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);
p_bou = mapminmax('apply', P_bou, ps_input);

[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);
t_bou = mapminmax('apply', T_bou, ps_output);

%%  ��������
% ���ز���Ԫ����5�����ж�����ز���Ϊ
% net = newff(p_train, t_train, [5,5]);
net = newff(p_train, t_train, 5);

%%  ����ѵ������
net.trainParam.epochs = 1000;     % �������� 
net.trainParam.goal = 1e-6;       % �����ֵ
net.trainParam.lr = 0.01;         % ѧϰ��

%%  ѵ������
net = train(net, p_train, t_train);
% ѵ�����net�п��Բ���Ȩ�غ�ƫ�õ���Ϣ
% Ȩ�ؾ���  net.IW{1}

%%  �������
t_sim1 = sim(net, p_train);
t_sim2 = sim(net, p_test);
t_sim3 = sim(net, p_bou);

%%  ���ݷ���һ��
T_sim1 = mapminmax('reverse', t_sim1, ps_output);
T_sim2 = mapminmax('reverse', t_sim2, ps_output);
T_sim3 = mapminmax('reverse', t_sim3, ps_output);

%%  ���������
error1 = sqrt(sum((T_sim1 - T_train).^2) ./ M);
error2 = sqrt(sum((T_sim2 - T_test ).^2) ./ N);
error3 = sqrt(sum((T_sim3 - T_bou ).^2) ./ B);

%%  ��ͼ
% RMSE��Ϊ���������
figure
plot(1: M, T_train, 'r-*', 1: M, T_sim1, 'b-o', 'LineWidth', 1)
legend('��ʵֵ','Ԥ��ֵ')
xlabel('Ԥ������')
ylabel('Ԥ����')
string = {'ѵ����Ԥ�����Ա�'; ['RMSE=' num2str(error1)]};
title(string)
xlim([1, M])
grid

figure
plot(1: N, T_test, 'r-*', 1: N, T_sim2, 'b-o', 'LineWidth', 1)
legend('��ʵֵ','Ԥ��ֵ')
xlabel('Ԥ������')
ylabel('Ԥ����')
string = {'���Լ�Ԥ�����Ա�';['RMSE=' num2str(error2)]};
title(string)
xlim([1, N])
grid

figure
plot(1: B, T_bou, 'r-*', 1: B, T_sim3, 'b-o', 'LineWidth', 1)
legend('��ʵֵ','Ԥ��ֵ')
xlabel('Ԥ������')
ylabel('Ԥ����')
string = {'Bouck��Ԥ�����Ա�';['RMSE=' num2str(error3)]};
title(string)
xlim([1, B])
grid

% �����ͬ��ԭ��
% 1.���ݾ������������
% 2.BP������ĳ�ʼȨ��������ģ����ܻ�Խ������Ӱ��
%%  ���ָ�����
% ����ϵ�� R2
R1 = 1 - norm(T_train - T_sim1)^2 / norm(T_train - mean(T_train))^2;
R2 = 1 - norm(T_test -  T_sim2)^2 / norm(T_test -  mean(T_test ))^2;
R3 = 1 - norm(T_bou -  T_sim3)^2 / norm(T_bou -  mean(T_bou ))^2;

disp(['ѵ�������ݵ�R2Ϊ��', num2str(R1)])
disp(['���Լ����ݵ�R2Ϊ��', num2str(R2)])
disp(['Bouck�����ݵ�R2Ϊ��', num2str(R3)])

% ƽ��������� MAE
mae1 = sum(abs(T_sim1 - T_train)) ./ M ;
mae2 = sum(abs(T_sim2 - T_test )) ./ N ;
mae3 = sum(abs(T_sim3 - T_bou )) ./ B ;

disp(['ѵ�������ݵ�MAEΪ��', num2str(mae1)])
disp(['���Լ����ݵ�MAEΪ��', num2str(mae2)])
disp(['Bouck�����ݵ�MAEΪ��', num2str(mae3)])

% ƽ����ԣ�ӦΪƫ���� MBE
mbe1 = sum(T_sim1 - T_train) ./ M ;
mbe2 = sum(T_sim2 - T_test ) ./ N ;
mbe3 = sum(T_sim3 - T_bou ) ./ B ;

disp(['ѵ�������ݵ�MBEΪ��', num2str(mbe1)])
disp(['���Լ����ݵ�MBEΪ��', num2str(mbe2)])
disp(['Bouck�����ݵ�MBEΪ��', num2str(mbe3)])