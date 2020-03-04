samples = [100];
%samples = [500];
q = 0.05;
Tau2 = 1;
Sigma2 = 1;
nrep = 10;
nmc = 2*10^2; 
start_measure = 'prior';
plotting_1 = false;
plotting_2 = false;
predictors = [1000];
%predictors = [250];

tic;
end_diff = zeros(length(predictors), nrep);
average_diff = zeros(length(samples), length(predictors));

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
                %Sample j uniformly on integers from 1 to p
                 for i=1:p
                     gamma_zero = gamma;
                     gamma_zero(i)=0;
                     gamma_one = gamma;
                     gamma_one(i)=1;
                     M_zero = logml(XX,Xy,yy,gamma_zero,Tau2,n);
                     log_zero_prior = log_pi_gamma(q,gamma_zero);
                     M_one = logml(XX,Xy,yy,gamma_one,Tau2,n);
                     log_one_prior = log_pi_gamma(q,gamma_one);
                     f_zero = M_zero + log_zero_prior;
                     f_one = M_one + log_one_prior;
                     p_one=1/(1+exp(f_zero - f_one));
                     choose_one = binornd(1,p_one);
                     if choose_one==1
                         gamma(i)=1;
                     else
                         gamma(i)=0;
                     end
                     %prop_gamma = gamma;
                     %if gamma(i)==1
                     %    prop_gamma(i)=0;
                     %else
                     %    prop_gamma(i)=1;
                     %end
                     %Mprop = logml(XX,Xy,yy,prop_gamma,Tau2,n);
                     %log_prop_prior = log(pi_gamma(q,prop_gamma));
                     %Mcurr = logml(XX,Xy,yy,gamma,Tau2,n);
                     %log_curr_prior = log(pi_gamma(q,gamma));
                     %Do this probabilistically
                     %if (Mprop + log_prop_prior > Mcurr + log_curr_prior)
                     %    disp('switch');
                     %    gamma=prop_gamma;
                     %end
                 end
                 gamma_array(:,t)=gamma;
                 mc_error(t) = mpm_err(gamma_array, GammaTrue,p,t); %get rid of this for speed
                 
            end  
            
            if(plotting_1)
                figure;
                plot(mc_error);
                xlabel('Iteration');
                ylabel('Normalized Error');
                title('Normalized Error of MPM Estimator Through Iterations');
            end
            
            %normalized_diff(r,ps) = mpm_err(gamma_array, GammaTrue);
            %if (plotting)
            %    figure;
            %    plot(normalized_diff);
            %end
            end_diff(ps,r)=mc_error(nmc);
        end
    end
    average_diff(ns,:)=transpose(mean(end_diff,2));
end

if(plotting_2)
    for i = 1:length(samples)
        figure;
        plot(predictors, average_diff(i,:));
        xlabel('Dimension');
        ylabel('Normalized Error');
        title('Normalized Error of MPM Estimator with Varying Dimension');
    end

    for j = 1:length(predictors)
        figure;
        plot(samples, average_diff(:,j));
        xlabel('Sample Size');
        ylabel('Normalized Error');
        title('Normalized Error of MPM Estimator with Varying Sample Size');
    end
end

function error = mpm_err(gamma_array, GammaTrue, p, t)
        gamma_totals = sum(gamma_array,2);
        median_model = gamma_totals >= (t/2);
        diff = sum(median_model ~= GammaTrue);
        error = diff/p;
end

function mode_error = mode_err(gamma_array, GammaTrue, p, t)
    curr = gamma_array(:,1:t);
    mode = findMode(curr);
    diff = sum(mode ~= GammaTrue);
    error = diff/p;
end

function prior = pi_gamma(q,gamma)
    p = length(gamma);
    size = sum(gamma);
    prior = q^size*(1-q)^(p-size);
end

function log_prior = log_pi_gamma(q,gamma)
    p = length(gamma);
    size = sum(gamma);
    prior = size*log(q) + (p-size)*log(1-q);
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

function fn = false_neg(guess, truth)
    fn = sum(guess < truth);
end

function fp = false_pos(guess, truth)
    fp=sum(guess>truth);
end
        
