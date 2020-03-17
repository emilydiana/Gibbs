%samples = [50, 100, 150, 200];
samples = [10];
q = 0.05;
Tau2 = 1;
Sigma2 = 1;
nrep = 1;
nmc = 2*10^5; 
interval = 100;
start_measure = 'prior';
plotting_1 = true;
plotting_2 = false;
%predictors = [50, 100, 150, 200, 250];
predictors = [10];

tic;
%end_diff = zeros(length(predictors), nrep);
end_mode_diff = zeros(length(predictors), nrep);
end_med_diff = zeros(length(predictors), nrep);
end_mode_fp = zeros(length(predictors), nrep);
end_med_fp = zeros(length(predictors), nrep);
%average_diff = zeros(length(samples), length(predictors));
average_med_diff = zeros(length(samples), length(predictors));
average_mode_diff = zeros(length(samples), length(predictors));
fp_med = zeros(length(samples), length(predictors));
fp_mode= zeros(length(samples), length(predictors));

for ns = 1:length(samples)
    n = samples(ns);
    for ps=1:length(predictors)
        p = predictors(ps);
        for r=1:nrep
            disp(['p = ' num2str(p) ' rep = ' num2str(r)]);
            toc; tic;
            %Let X be identity???
            X = normrnd(0,1,[n p]);
            Beta = zeros(p,1);
            s = binornd(p,q);
            Beta(1:s) = normrnd(0,sqrt(Tau2),[s 1]);

            y = X*Beta + normrnd(0,sqrt(Sigma2),[n 1]);
            
            [B, FitInfo] = lasso(X,y,'CV',10);
            lassoPlot(B,FitInfo,'PlotType','CV');
            legend('show') % Show legend
            idxLambda1SE = FitInfo.Index1SE;
            coef = B(:,idxLambda1SE);
            %coef0 = FitInfo.Intercept(idxLambda1SE); % do we have an
            %intercept??? I think no

            yy = y'*y;
            XX = X'*X;
            Xy = X'*y;

            if strcmp(start_measure,'prior')
                gamma = binornd(1,q,[p 1]);
            end
            if strcmp(start_measure,'null')
                gamma = zeros(p,1);
            end

            GammaTrue = zeros(p,1);
            GammaTrue(1:s) = 1;
            gamma_array = zeros(p,nmc);
            mpm_error = zeros(floor(nmc/interval),1); %can do this every t iterations later for speed
            mode_error = zeros(floor(nmc/interval),1);
            mpm_fp = zeros(floor(nmc/interval),1); %can do this every t iterations later for speed
            mode_fp = zeros(floor(nmc/interval),1);
            
            M_curr = logml(XX,Xy,yy,gamma,Tau2,n);
            prior_curr = log_pi_gamma(q,gamma);
            f_curr = M_curr + prior_curr;
            
            for t = 1:nmc
                i = randsample(p,1);              
                gamma_zero = gamma;
                gamma_zero(i)=0;
                gamma_one = gamma;
                gamma_one(i)=1;
                f_zero = f_curr;
                f_one = f_curr;
                if (gamma_zero == gamma)
                    M_one = logml(XX,Xy,yy,gamma_one,Tau2,n);
                    log_one_prior = log_pi_gamma(q,gamma_one);
                    f_one = M_one + log_one_prior;
                else
                    M_zero = logml(XX,Xy,yy,gamma_zero,Tau2,n);
                    log_zero_prior = log_pi_gamma(q,gamma_zero);
                    f_zero = M_zero + log_zero_prior;
                end
                p_one=1/(1+exp(f_zero - f_one));
                choose_one = binornd(1,p_one);
                if choose_one==1
                    gamma(i)=1;
                else
                    gamma(i)=0;
                end
                gamma_array(:,t)=gamma;
                    %[mpm_error(t), mpm_fp(t)] = mpm_err(gamma_array, GammaTrue,p,t); %get rid of this for speed
                    %[mode_error(t), mode_fp(t)] = mode_err(gamma_array, GammaTrue,p,t);
            end
            
            %if(plotting_1)
            %    figure;
            %    hold on;
            %    p1 = plot(mpm_error);
            %    p2 = plot(mode_error);
            %    p3 = plot(mpm_fp);
            %    p4 = plot(mode_fp);
            %    xlabel('Iteration');
            %    ylabel('Normalized Error');
            %    title('Normalized Error of MPM Estimator Through Iterations');
            %    legend([p1,p2,p3,p4],{'Error MPM','Error HPM','FP MPM','FP HPM'});
            %end
            
            %normalized_diff(r,ps) = mpm_err(gamma_array, GammaTrue);
            %if (plotting)
            %    figure;
            %    plot(normalized_diff);
            %end
            %end_diff(ps,r)=mpm_error(nmc);
            final_model = gamma_array(:,end);
            for j = 1: length(mpm_error)
               [mpm_error(j), mpm_fp(j)] = mpm_err(gamma_array, final_model,p,j*interval); %get rid of this for speed
               [mode_error(j), mode_fp(j)] = mode_err(gamma_array, final_model,p,j*interval);
            end
            
            if(plotting_1)
                figure;
                hold on;
                p1 = plot(mpm_error);
                p2 = plot(mode_error);
                p3 = plot(mpm_fp);
                p4 = plot(mode_fp);
                xlabel('Iteration Interval');
                ylabel('Normalized Error');
                title('Normalized Error of MPM Estimator Through Iterations');
                legend([p1,p2,p3,p4],{'Error MPM','Error HPM','FP MPM','FP HPM'});
            end
            
            [end_med_diff(ps,r), end_med_fp(ps,r)] = mpm_err(gamma_array, final_model, p, nmc);
            [end_mode_diff(ps,r),  end_mode_fp(ps,r)] = mode_err(gamma_array, final_model, p, nmc);
        end
    end
    average_med_diff(ns,:)=transpose(mean(end_med_diff,2));
    average_mode_diff(ns,:)=transpose(mean(end_mode_diff,2));
    fp_med(ns,:)=transpose(mean(end_med_fp,2));
    fp_mode(ns,:)=transpose(mean(end_mode_fp,2));
end

if(plotting_2)
    for i = 1:length(samples)
        n = samples(i);
        figure;
        hold on;
        p1 = plot(predictors, average_med_diff(i,:));
        p2 = plot(predictors, average_mode_diff(i,:));
        p3 = plot(predictors, fp_med(i,:));
        p4 = plot(predictors, fp_mode(i,:));
        xlabel('Dimension');
        ylabel('Normalized Error');
        title(sprintf('Error with %d Samples and Varying Dimension',n));
        legend([p1,p2,p3,p4],{'Error MPM','Error HPM','FP MPM', 'FP HPM'});
    end

    for j = 1:length(predictors)
        p = predictors(i);
        figure;
        hold on;
        p1 = plot(samples, average_med_diff(:,j));
        p2 = plot(samples, average_mode_diff(:,j));
        p3 = plot(samples, fp_med(:,j));
        p4 = plot(samples, fp_mode(:,j));
        xlabel('Sample Size');
        ylabel('Normalized Error');
        title(sprintf('Error with %d Predictors and Varying Sample Size',p));
        legend([p1,p2,p3,p4],{'Error MPM','Error HPM', 'FP MPM', 'FP HPM'});
    end
    
    
    
    %for i = 1:length(samples)
    %    figure;
    %    plot(predictors, average_mode_diff(i,:));
    %    xlabel('Dimension');
    %    ylabel('Normalized Error');
    %    title('Normalized Error of HPM Estimator with Varying Dimension');
    %end

    %for j = 1:length(predictors)
    %    figure;
    %    plot(samples, average_mode_diff(:,j));
    %    xlabel('Sample Size');
    %    ylabel('Normalized Error');
    %    title('Normalized Error of HPM Estimator with Varying Sample Size');
    %end
end


function [error, fp] = mpm_err(gamma_array, gamma_end, p, t)
        gamma_totals = sum(gamma_array,2);
        median_model = gamma_totals >= (t/2);
        diff = sum(median_model ~= gamma_end);
        fp=sum(median_model > gamma_end);
        fp=fp/p;
        error = diff/p;
end

function [mode_error, fp] = mode_err(gamma_array, gamma_end, p, t)
    curr = gamma_array(:,1:t);
    mode = findMode(curr);
    diff = sum(mode ~= gamma_end);
    fp=sum(mode>gamma_end);
    fp=fp/p;
    mode_error = diff/p;
end

function prior = pi_gamma(q,gamma)
    p = length(gamma);
    size = sum(gamma);
    prior = q^size*(1-q)^(p-size);
end

function log_prior = log_pi_gamma(q,gamma)
    p = length(gamma);
    size = sum(gamma);
    log_prior = size*log(q) + (p-size)*log(1-q);
end

function M = logml(XX,Xy,yy,gamma,Tau2,n)
    m = sum(gamma);
    if m==0
        M = gammaln(n/2)-(n/2).*log(2*pi)-(n/2).*log(yy/2);
    else
        XXg = XX(gamma==1,gamma==1);
        Xyg = Xy(gamma==1);
        Ig = eye(m);
        C = chol((Tau2.*XXg+Ig));
        Ci = chol((Tau2^(-1).*Ig+XXg));
        CiXyg = Ci'\Xyg;
        M = gammaln(n/2)-sum(log(sqrt(2*pi).*diag(C)))-(n/2).*log(.5.*(yy-(CiXyg'*CiXyg)));
    end
end

function mode_gamma=findMode(gamma_array)
    [d, ~]=size(gamma_array);
    gamma_decimal = bi2de(gamma_array');
    mode_decimal = mode(gamma_decimal);
    mode_gamma = de2bi(mode_decimal);
    mode_gamma(d)=0;
    mode_gamma = mode_gamma';
end

%function fn = false_neg(guess, truth,p)
%    fn = sum(guess < truth);
%    fn = fn/p;
%end

%function fp_mode = false_pos(gamma_array, truth,p)
%    guess = findMode(gamma_array);
%    fp=sum(guess>truth);
%    fp = fp/p;
%end

%function fp_median = false_pos(gamma_array, truth,p)
%    guess = findMed(gamma_array);
%    fp=sum(guess>truth);
%    fp = fp/p;
%end
        
