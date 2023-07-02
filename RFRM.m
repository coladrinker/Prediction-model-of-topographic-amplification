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
p_train = p_train'; p_test = p_test';
t_train = t_train'; t_test = t_test';

%%  
trees = 100;                               
leaf  = 5;                                  
OOBPrediction = 'on';                      
OOBPredictorImportance = 'on';           
Method = 'regression';                            
net = TreeBagger(trees, p_train, t_train, 'OOBPredictorImportance', OOBPredictorImportance,...
      'Method', Method, 'OOBPrediction', OOBPrediction, 'minleaf', leaf);
importance = net.OOBPermutedPredictorDeltaError;  

%%  
t_sim1 = predict(net, p_train);
t_sim2 = predict(net, p_test );

%%
T_sim1 = mapminmax('reverse', t_sim1, ps_output);
T_sim2 = mapminmax('reverse', t_sim2, ps_output);

%% 
error1 = sqrt(sum((T_sim1' - T_train).^2) ./ M);
error2 = sqrt(sum((T_sim2' - T_test ).^2) ./ N);

%% 
figure
plot(1: M, T_train, 'r-*', 1: M, T_sim1, 'b-o', 'LineWidth', 1)
legend('Real value','Predicted value')
xlabel('Sample')
ylabel('AR_{(h,max)}')
string = ['RMSE of RFRM=' num2str(error1)];
title(string)
xlim([1, M])
grid
set(gcf, 'unit', 'centimeters', 'position', [10 5 30 10]);

figure
plot(1: N, T_test, 'r-*', 1: N, T_sim2, 'b-o', 'LineWidth', 1)
legend('Real value','Predicted value')
xlabel('Sample')
ylabel('AR_{(h,max)}')
string = ['RMSE of RFRM=' num2str(error2)];
title(string)
xlim([1, N])
grid
set(gcf, 'unit', 'centimeters', 'position', [10 5 14 10]);

%%
figure
plot(1 : trees, oobError(net), 'b-', 'LineWidth', 1)
legend('Error Curve')
xlabel('Number of decision trees')
ylabel('Out of bag error')
xlim([1, trees])
grid

%%
figure
color_matrix = [0.85,0.325,0.098;0.92,0.694,0.125;0,0.447,0.741;0.466,0.674,0.188;0.494,0.184,0.556];  %每个柱子的颜色设置
b = bar(importance); 
b.FaceColor = 'flat';
for i=1:5
    b.CData(i,:) = color_matrix(i,:);
end
%
Xlabel = {'\itH','\itα','\itfr','\itλ','\itN'};
set(gca,'XTick',[1 2 3 4 5],'FontSize',12);
%
set(gca,'XTickLabel',Xlabel);
xlabel('Parameters')
ylabel('Importance')
grid on
box on
%%  
%  R2
R1 = 1 - norm(T_train - T_sim1')^2 / norm(T_train - mean(T_train))^2;
R2 = 1 - norm(T_test -  T_sim2')^2 / norm(T_test -  mean(T_test ))^2;

%  MAE
mae1 = sum(abs(T_sim1' - T_train)) ./ M;
mae2 = sum(abs(T_sim2' - T_test )) ./ N;

%  MBE
mbe1 = sum(T_sim1' - T_train) ./ M ;
mbe2 = sum(T_sim2' - T_test ) ./ N ;

% 
stdre1 = std(abs(T_sim1' - T_train)./T_train);
stdre2 = std(abs(T_sim2' - T_test)./T_test);

%%  
figure
scatter(T_train, T_sim1', 'o','b');
hold on
scatter(T_test, T_sim2', '^','r');
hold on
plot([1,1.7],[1,1.7],'--k','linewidth',2);
legend({'Training set','Test set','The 1:1 line'},'Location','northwest')
xlabel('Real value')
ylabel('Predicted value')
string = {'RFRM'};
title(string)
xlim([0.9, 1.8])
ylim([0.9, 1.8])
txt = {['R^2 of training set=' num2str(R1)];['R^2 of test set=' num2str(R2)]};
text(1.4,1.1,txt)
grid on
box on
set(gcf, 'unit', 'centimeters', 'position', [10 5 14 10]);

