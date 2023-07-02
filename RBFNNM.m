%%  
warning off          
close all            
clear               
clc                  

%%
res = readmatrix('Data-all.xlsx');

%%  
temp = randperm(182);
P_train = res([temp(1: 146)], 1: 5)';
T_train = res([temp(1: 146)], 6)';
M = size(P_train, 2);

P_test = res([temp(147: 182)], 1: 5)';
T_test = res([temp(147: 182)], 6)';
N = size(P_test, 2);

%%  
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);

[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);

%%  
rbf_spread = 10;                          
net = newrbe(p_train, t_train, rbf_spread);

%%  
t_sim1 = sim(net, p_train);
t_sim2 = sim(net, p_test );

%%  
T_sim1 = mapminmax('reverse', t_sim1, ps_output);
T_sim2 = mapminmax('reverse', t_sim2, ps_output);

%% 
error1 = sqrt(sum((T_sim1 - T_train).^2) ./ M);
error2 = sqrt(sum((T_sim2 - T_test ).^2) ./ N);

%%  
view(net)

%% 
figure
plot(1: M, T_train, 'r-*', 1: M, T_sim1, 'b-o', 'LineWidth', 1)
legend('Real value','Predicted value')
xlabel('Sample')
ylabel('AR_{(h,max)}')
string = ['RMSE of RBFNNM=' num2str(error1)];
title(string)
xlim([1, M])
grid
set(gcf, 'unit', 'centimeters', 'position', [10 5 30 10]);

figure
plot(1: N, T_test, 'r-*', 1: N, T_sim2, 'b-o', 'LineWidth', 1)
legend('Real value','Predicted value')
xlabel('Sample')
ylabel('AR_{(h,max)}')
string = ['RMSE of RBFNNM=' num2str(error2)];
title(string)
xlim([1, N])
grid
set(gcf, 'unit', 'centimeters', 'position', [10 5 14 10]);


%%  
%  R2
R1 = 1 - norm(T_train - T_sim1)^2 / norm(T_train - mean(T_train))^2;
R2 = 1 - norm(T_test -  T_sim2)^2 / norm(T_test -  mean(T_test ))^2;

%  MAE
mae1 = sum(abs(T_sim1 - T_train)) ./ M ;
mae2 = sum(abs(T_sim2 -  T_test)) ./ N ;

%  MBE
mbe1 = sum(T_sim1 - T_train) ./ M ;
mbe2 = sum(T_sim2 -  T_test) ./ N ;

% 
stdre1 = std(abs(T_sim1 - T_train)./T_train);
stdre2 = std(abs(T_sim2 - T_test)./T_test);

%%  
figure
scatter(T_train, T_sim1, 'o','b');
hold on
scatter(T_test, T_sim2, '^','r');
hold on
plot([1,1.7],[1,1.7],'--k','linewidth',2);
legend({'Training set','Test set','The 1:1 line'},'Location','northwest')
xlabel('Real value')
ylabel('Predicted value')
string = {'RBFNNM'};
title(string)
xlim([0.9, 1.8])
ylim([0.9, 1.8])
txt = {['R^2 of training set=' num2str(R1)];['R^2 of test set=' num2str(R2)]};
text(1.4,1.1,txt)
grid on
box on
set(gcf, 'unit', 'centimeters', 'position', [10 5 14 10]);
