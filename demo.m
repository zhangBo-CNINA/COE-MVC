clear;
clc;
warning off;
addpath ./measure/
addpath ./functions/
% COE-MVC: Continual Multi-View Clustering in Open Environments
%% dataset
ds = {'Fashion'};
dsPath  = './data/';
resPath = './res/';

for dsi = 1:length(ds)
    % load data & make folder
    dataName = ds{dsi}; disp(dataName);
    load_path = ['./data/', dataName, '_Decaying.mat'];
    %load_path = ['./data/', dataName, '_Uniform Distribution.mat'];
    load(load_path);
    k = length(unique(Y));
    n = length(Y);
    V = length(Xnew0f);
    %matpath = strcat(resPath,dataName);
    %txtpath = strcat(resPath,strcat(dataName,'.txt'));
   
    %dlmwrite(txtpath, strcat('Dataset:',cellstr(dataName), '  Date:',datestr(now)),'-append','delimiter','','newline','pc');    

    

    %% param setting
    perf_std=[];perf_accu =[];MEAN_perf=[];
    param.k = k;
    LM = [1e-4,1e-3,1e-2,1e-1, 1e0, 1e1, 1e2, 1e3, 1e4];
    MU = [1e-4,1e-3,1e-2,1e-1, 1e0, 1e1, 1e2, 1e3, 1e4];
    GM = [1e-4,1e-3,1e-2,1e-1, 1e0, 1e1, 1e2, 1e3, 1e4];
    for rp1 = 1%1:length(LM)
        for rp2 = 5%1:length(MU)
            for rp3 = 3%1:length(GM)
            param.lambda = LM(rp1);
            param.mu     = MU(rp2);
            param.gamma  = GM(rp3);
           
     flag = 0; perf = []; L =[];
     all_losses = cell(V, 1);
    tic
    for t = 1:V
        Xt  = Xnew0f{t};
        nt = size(Xt,2);
     
        if t == 1
             [Wt, At, Zt,ns, loss] = Init_COE_MVC(Xt, param);  
             A_old = At;
             W_old = Wt;       
             Z_old = Zt;  
             n_old = ns;
             all_losses{t} = loss;
        else
            [Wt, At, Zt,ns,Z_old, loss] = Inc_COE_MVC(Xt, W_old, A_old,Z_old,n_old, param);
             A_old = At;
             W_old = Wt; 
             if ns>=n_old
                
                Z_old = Zt;  
             else
               Zt = Z_old;
             end
             n_old = ns;
             all_losses{t} = loss;
        end
       
    end  
         
    [~, res_all] = myNMIACCwithmean(Zt',Y,k); %[ACC nmi Purity Fscore Precision Recall AR Entropy];
    time = toc;

              % if(res_all(1)>0.25)
            
                fprintf("\n ITER: (%d, %d, %d) acc: %.4f, nmi:%.4f, p:%.4f, f:%.4f, AR:%.4f, time:%.4f", rp1, rp2, rp3, res_all(1), res_all(2), res_all(3), res_all(4), res_all(7),time);
               %  end 


  
                figure('Name', 'Empirical Convergence Analysis', 'Position', [100, 100, 600, 450]);
                hold on;
                
                color_map = lines(V); 
                legend_labels = cell(V, 1);
                
                for t = 1:V
               
                    current_loss = all_losses{t};
                  
                    plot(1:length(current_loss), current_loss, '-o', ...
                        'LineWidth', 1.5, ...
                        'MarkerSize', 5, ...
                        'Color', color_map(t, :), ...
                        'MarkerFaceColor', color_map(t, :));
                    
                   
                    legend_labels{t} = sprintf('Round %d', t);
                end
                
                hold off;
                grid on;
                
               
                xlabel('Iteration', 'FontSize', 12, 'FontWeight', 'bold');
                ylabel('Objective Function Value', 'FontSize', 12, 'FontWeight', 'bold');
                title(['COE-MVC-D'], 'FontSize', 14, 'FontWeight', 'bold');
                
            
                legend(legend_labels, 'Location', 'northeast', 'FontSize', 10);
              
                set(gca, 'FontSize', 11, 'LineWidth', 1);
            
            end
        end
    end
    clear X Ynew k Z Wt  At
end 
%[a,b] = max(perf_accu(:,4))
%result = [ACC nmi Purity Fscore Precision Recall AR Entropy];