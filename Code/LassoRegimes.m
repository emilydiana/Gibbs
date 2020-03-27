samples = [100];
q = 0.05;
Tau2 = 1;
Sigma2 = 1;
nrep = 10;
nmc = 2*10^5; 
predictors = [100:100:1000];

tic;

for ns = 1:length(samples)
    n = samples(ns);
    mse = zeros(length(predictors),1);
    for ps=1:length(predictors)
        p = predictors(ps);
        for r=1:nrep
            mse_loc = zeros(nrep,1);
            disp(['p = ' num2str(p) ' rep = ' num2str(r)]);
            toc; tic;
            X = normrnd(0,1,[n p]);             %Let X be identity???
            Beta = zeros(p,1);
            s = binornd(p,q);
            Beta(1:s) = normrnd(0,sqrt(Tau2),[s 1]);

            y = X*Beta + normrnd(0,sqrt(Sigma2),[n 1]);
            
            [B, FitInfo] = lasso(X,y,'CV',10);
            lassoPlot(B,FitInfo,'PlotType','CV');
            legend('show') % Show legend
            idxLambda1SE = FitInfo.Index1SE;
            coef = B(:,idxLambda1SE);
            mse_loc(r)= FitInfo.MSE(idxLambda1SE);
            %coef0 = FitInfo.Intercept(idxLambda1SE); % do we have an
            %intercept??? I think no
        end
        mse(ps)= mean(mse_loc);
    end
end


figure;
plot(predictors, mse);
xlabel('Predictors');
ylabel('MSE');
title('Error with 100 Samples and Varying Number of Predictors');