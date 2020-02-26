n = 1000;
q = 0.05;
Tau2 = 1;
Sigma2 = 1;
nrep = 10;
nmc = 2*10^2; 
start_measure = 'prior';

predictors = [50, 100, 150, 200, 250];

tic;
end_diff = zeros(length(predictors), nrep);

for ps=1:length(predictors)
    p = predictors(ps);
    for r=1:nrep
        disp(['p = ' num2str(p) ' rep = ' num2str(r)]);
        toc; tic;
        
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
        mc_error = zeros(nmc,1);
        for t = 1:nmc
             for i=1:p
                 prop_gamma = gamma;
                 if gamma(i)==1
                     prop_gamma(i)=0;
                 else
                     prop_gamma(i)=1;
                 end
                 Mprop = logml(XX,Xy,yy,prop_gamma,Tau2,n);
                 log_prop_prior = log(pi_gamma(q,prop_gamma));
                 Mcurr = logml(XX,Xy,yy,gamma,Tau2,n);
                 log_curr_prior = log(pi_gamma(q,gamma));
                 if (Mprop + log_prop_prior > Mcurr + log_curr_prior)
                     disp('switch');
                     gamma=prop_gamma;
                 end
             end
             gamma_array(:,t)=gamma;
             mc_error(t) = mpm_err(gamma_array, GammaTrue);
        end  
        %figure;
        %plot(mc_error);
        
        %normalized_diff(r,ps) = mpm_err(gamma_array, GammaTrue);
        %if (plotting)
        %    figure;
        %    plot(normalized_diff);
        %end
        end_diff(ps,r)=mc_error(nmc);
    end
end

average_diffs=mean(end_diff,2);

figure;
plot(predictors, average_diffs);
xlabel('Dimension');
ylabel('Normalized Error');
title('Normalized Error of MPM Estimator');


function error = mpm_err(gamma_array, GammaTrue)
        gamma_totals = sum(gamma_array,2);
        [pred,iter] = size(gamma_array); 
        median_model = gamma_totals >= (iter/2);
        diff = sum(median_model ~= GammaTrue);
        error = diff/pred;
end

function prior = pi_gamma(q,gamma)
    p = length(gamma);
    size = sum(gamma);
    prior = q^size*(1-q)^(p-size);
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


