%data = readmatrix('Ozone.txt');
opts = detectImportOptions('Ozone.csv');
preview('Ozone.csv',opts);

data = readmatrix('Ozone.csv', opts);

%%%%
X=data(:,3:37);
y=data(:,2);
Tau2=1;
nmc = 2*10^5;
q=0.05;
p=35;
n=178;
random_update=true;
%%%%%

yy = y'*y;
XX = X'*X;
Xy = X'*y;
            
gamma = binornd(1,q,[p 1]);
gamma_array = zeros(p,nmc);                


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
