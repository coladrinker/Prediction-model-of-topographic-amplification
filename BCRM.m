%%  
warning off           
close all               
clear                  
clc                  
%% 
res = readmatrix('BCRM-Data-all.xlsx');

%%  
T_train = res(1:187, 6)';
M = size(T_train, 2);

%%  
T_sim1 =  res(1:187, 8)';

%%  
error1 = sqrt(sum((T_sim1 - T_train).^2) ./ M);

%%  
% 
figure
plot(1: M, T_train, 'r-*', 1: M, T_sim1, 'b-o', 'LineWidth', 1)
legend('Real value','Predicted value')
xlabel('Sample')
ylabel('AR_{(h,max)}')
string = ['RMSE of BCRM=' num2str(error1)];
title(string)
xlim([1, M])
grid
set(gcf, 'unit', 'centimeters', 'position', [10 5 30 10]);

%%  
% R2
R1 = 1 - norm(T_train - T_sim1)^2 / norm(T_train - mean(T_train))^2;

disp(['R2 is£º', num2str(R1)])

% MAE
mae1 = sum(abs(T_sim1 - T_train)) ./ M ;

disp(['MAE is£º', num2str(mae1)])

% MBE
mbe1 = sum(T_sim1 - T_train) ./ M ;

disp(['MBE is£º', num2str(mbe1)])

% 
stdre1 = std(abs(T_sim1 - T_train)./T_train);

disp(['Std is£º', num2str(stdre1)])

%% 
figure
scatter(T_train, T_sim1, 'o','b');
hold on
plot([1,1.7],[1,1.7],'--k','linewidth',2);
legend({'Samples in Data-all','The 1:1 line'},'Location','northwest')
xlabel('Real value')
ylabel('Predicted value')
string = {'BCRM'};
title(string)
xlim([0.9, 1.8])
ylim([0.9, 1.8])
txt = {['R^2 of BCRM=' num2str(R1)]};
text(1.4,1.1,txt)
grid on
box on
set(gcf, 'unit', 'centimeters', 'position', [10 5 14 10]);
