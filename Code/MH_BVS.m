samples = [100];
q = 0.75;
Tau2 = 4;
Sigma2 = 1;
nrep = 1;
nmc = 2*10^5; 
interval = 1000; %Number of MC iterations between each plotting point
start_measure = 'prior';
random_update=true;
plotting_1 = true;
plotting_2 = false;
plotting_3 = false;
predictors = [10 50 100 200 400 600 800 1000];
%predictors = [100:100:1000];
epsilon = 0.001;

tic;

%Preallocated arrays for plotting
end_mode_diff = zeros(length(predictors), nrep);
end_med_diff = zeros(length(predictors), nrep);
end_mode_fp = zeros(length(predictors), nrep);
end_med_fp = zeros(length(predictors), nrep);
mse_diff = zeros(length(predictors), nrep);
med_size = zeros(length(predictors), nrep);
mode_size = zeros(length(predictors), nrep);
prob_one = zeros(length(predictors), nrep);
prob_two = zeros(length(predictors), nrep);
prob_three = zeros(length(predictors), nrep);
size_one = zeros(length(predictors), nrep);
size_two = zeros(length(predictors), nrep);size_three = zeros(length(predictors), nrep);
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
            %Add correlation?
            %Sigma = rho.^abs(bsxfun(@minus,(1:p)',(1:p)))
            
            X = normrnd(0,1,[n p]); 
            Beta = zeros(p,1);
            s = binornd(p,q);
            Beta(1:s) = normrnd(0,sqrt(Tau2),[s 1]);

            y = X*Beta + normrnd(0,sqrt(Sigma2),[n 1]);

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
            
            %Allocate more plotting arrays
            mpm_error = zeros(floor(nmc/interval),1); 
            mode_error = zeros(floor(nmc/interval),1);
            mpm_fp = zeros(floor(nmc/interval),1); 
            mode_fp = zeros(floor(nmc/interval),1);
            
            M_curr = logml(XX,Xy,yy,gamma,Tau2,n);
            prior_curr = log_pi_gamma(q,gamma);
            f_curr = M_curr + prior_curr;
            i=1;
            for t = 1:nmc
                if random_update
                    i = randsample(p,1);    
                else
                    i = mod(i+1,p);
                end
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
            end

            final_model_med  = findMed(gamma_array, nmc);
            final_model_mode = findMode(gamma_array);
            
            for j = 1: length(mpm_error)
               [mpm_error(j), mpm_fp(j)] = mpm_err(gamma_array, final_model_med,p,j*interval); %get rid of this for speed
               [mode_error(j), mode_fp(j)] = mode_err(gamma_array, final_model_mode,p,j*interval);
            end
            
            if(plotting_1)
                figure;
                hold on;
                p1 = plot(mpm_error);
                p2 = plot(mode_error);
                p3 = plot(mpm_fp);
                p4 = plot(mode_fp);
                p5 = xline(conv_ind(mpm_error,epsilon,nmc),'--');
                p6 = xline(conv_ind(mode_error,epsilon,nmc),'-');
                xlabel('Iteration Interval');
                ylabel('Normalized Error');
                title(sprintf('Normalized Error with %d Predictors and Sample Size 100',p));
                legend([p1,p2,p3,p4,p5,p6],{'Error MPM','Error HPM','FP MPM','FP HPM','MPM Conv','HPM Conv'});
            end
            
            [end_med_diff(ps,r), end_med_fp(ps,r)] = mpm_err(gamma_array, final_model_med, p, nmc);
            [end_mode_diff(ps,r),  end_mode_fp(ps,r)] = mode_err(gamma_array, final_model_mode, p, nmc);
            [bma_model, bma_mse] = bma(gamma_array,Tau2,Sigma2,XX,Xy,X,y);
            med_size(ps,r) = sum(final_model_med);
            mode_size(ps,r) = sum(final_model_mode);

            %compare against lasso
            [B, FitInfo] = lasso(X,y,'CV',10);
            idxLambda1SE = FitInfo.Index1SE;
            lasso_model = B(:,idxLambda1SE);
            lasso_mse = FitInfo.MSE(idxLambda1SE);
            mse_diff(ps,r) = lasso_mse - bma_mse;
            
            [one, one_prob(ps,r), two, two_prob(ps,r), three, three_prob(ps,r)] = findMode3(gamma_array,XX,Xy,yy,Tau2,n,q);
            size_one(ps,r) = sum(one);
            size_two(ps,r) = sum(two);
            size_three(ps,r) = sum(three);
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
end

if(plotting_3)
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
end  

%Calculate median
function med_gamma = findMed(gamma_array, n)
        gamma_totals = sum(gamma_array(:,[1:n]),2);
        med_gamma = gamma_totals >= (n/2);
end

%Calculate mode
function mode_gamma=findMode(gamma_array)
    [d, ~]=size(gamma_array);
    gamma_decimal = bi2de(gamma_array');
    mode_decimal = mode(gamma_decimal);
    mode_gamma = de2bi(mode_decimal);
    if(length(mode_gamma)<d)
        mode_gamma(d)=0;
    end
    mode_gamma = mode_gamma';
end

function [first, first_prob, second, second_prob, third, third_prob] = findMode3(gamma_array,XX,Xy,yy,Tau2,n,q)
    [d, w]=size(gamma_array);
    gamma_decimal = bi2de(gamma_array');
    [m, bin] = hist(gamma_decimal,unique(gamma_decimal));
    [~,idx] = sort(-m);
    top_3 = bin(idx);
    freq = m(idx);
    first = de2bi(top_3(1))';
    %M_first = logml(XX,Xy,yy,first,Tau2,n);
    %log_first_prior = log_pi_gamma(q,first);
    %first_prob = exp(M_first + log_first_prior); 
    first_prob = freq(1)/w;
    second = de2bi(top_3(2))';
    %M_second = logml(XX,Xy,yy,second,Tau2,m);
    %log_second_prior = log_pi_gamma(q,second);
    %second_prob = exp(M_second + log_second_prior);
    second_prob = freq(2)/w;
    third = de2bi(top_3(3))';
    %M_third = logml(XX,Xy,yy,third,Tau2,m);
    %log_third_prior = log_pi_gamma(q,third);
    %third_prob = exp(M_third + log_third_prior);
    third_prob = freq(3)/w;
end

%Calculate error and false positive rate for MPM compared to empirical
%limit
function [error, fp] = mpm_err(gamma_array, gamma_end, p, t)
        median_model = findMed(gamma_array,t);
        diff = sum(median_model ~= gamma_end);
        fp=sum(median_model > gamma_end);
        fp=fp/p;
        error = diff/p;
end

%Calculate error and false positive rate for HPM compared to empirical
%limit
function [mode_error, fp] = mode_err(gamma_array, gamma_end, p, t)
    curr = gamma_array(:,1:t);
    mode = findMode(curr);
    diff = sum(mode ~= gamma_end);
    fp=sum(mode>gamma_end);
    fp=fp/p;
    mode_error = diff/p;
end

%Calculate iteration at which convergence has been reached
function ind = conv_ind(errors, epsilon,nmc)
   temp = errors >= epsilon;
   ind = find(temp, 1, 'last');
   ind = min(ind + 1, nmc);
end

%Calculate log prior for gamma vector
function log_prior = log_pi_gamma(q,gamma)
    p = length(gamma);
    size = sum(gamma);
    log_prior = size*log(q) + (p-size)*log(1-q);
end

%Gaussian log likelihood
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

function model=beta_post(gamma,Tau2,Sigma2,XX,Xy)
    m=length(gamma);
    gamma_ind = gamma*gamma';
    XXg = XX.*gamma_ind;
    Xyg = Xy.*gamma;
    Ig = eye(m);
    model = inv(Tau2.*XXg + Sigma2.*Ig)*Xyg;
end

function [bma_model, mse] = bma(gamma_array,Tau2,Sigma2,XX,Xy,X,y)
  [pred, nm] = size(gamma_array);
  betas = zeros(pred,nm);
  for i = 1:nm
      betas(:,i) = beta_post(gamma_array(:,i),Tau2,Sigma2,XX,Xy);
  end
  bma_model = mean(betas,2);
  mse =  mean(X*bma_model - y).^2;
end
