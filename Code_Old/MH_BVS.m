n = 100;
q = 0.05;
Tau2 = 1;
Sigma2 = 1;
sigma_squared = Sigma2;
p_single = 0.9;
disp_int = 10000;
%nrep = 50;
nrep = 1;
nmc = 1; %Number of monte carlo iterations??
start_measure = 'null';
saving = false;
plotting = true;
lambda=1;
%ps = 2200:200:3000;
ps = 2200;

tic;
for p=ps
    for r=1:nrep
        disp(['p = ' num2str(p) ' rep = ' num2str(r)]);
        toc; tic;
        
        X = normrnd(0,1,[n p]);
        XX = X'*X;
        X_inv = pinv(XX);
        %Will this error out sometimes?
        Beta = zeros(p,1);
        s = binornd(p,q);

        %var = sqrt(Tau2)*X_;
        %Beta = normrnd(0,sqrt(Tau2).*X_inv);
        %How should I do this given gamma?
        %ED: To do
        Beta(1:s) = normrnd(0,sqrt(Sigma2),[s 1]);
        %Beta(1:s) = normrnd(0,sqrt(Tau2).*X_inv,[s 1]);
        %Setting tau2 and sigma2 equal
        
        %Beta(1:s) = normrnd(0,sqrt(Tau2),[s 1]);
        %This needs to be different for gibbs sampling setup
        %Sigma squared or tau squared?
        y = X*Beta + normrnd(0,sqrt(Sigma2),[n 1]);
        
        yy = y'*y;
        Xy = X'*y;
        B_ols = y\X;
        
        if strcmp(start_measure,'prior')
            gamma1 = binornd(1,q,[p 1]); gamma2 = zeros(p,1);
        end
        if strcmp(start_measure,'null')
            gamma1 = zeros(p,1); gamma2 = zeros(p,1);
        end
        mgamma1 = sum(gamma1); mgamma2 = sum(gamma2);
        
        GammaTrue = zeros(p,1);
        GammaTrue(1:s) = 1;
        
        D01 = zeros(nmc,1); D02 = zeros(nmc,1);
        %GAMMA = zeros(nmc,p);
        ACC1 = zeros(nmc,1); ACC2 = zeros(nmc,1);
        TP1 = zeros(nmc,1); TP2 = zeros(nmc,1);
        FN1 = zeros(nmc,1); FN2 = zeros(nmc,1);
        D12 = zeros(nmc,1);
        
        MGAMMA1 = zeros(nmc,1);
        LBF1 = zeros(nmc,1);
        
        %Here is where the MCMC happens
        for t = 1:nmc
            %draw beta
            %Beta(1:s) = normrnd(0,sqrt(Tau2),[s 1]);
            A = pinv(sigma_squared^(-2).*XX + X_inv);
            beta = normrnd((sigma_squared)^(-2).*A*XX*B_ols',A);
            %Put in D here
            %draw sigma
            %sigma_squared_inv = gamrnd(nu/2 ,nu*lambda/2);
            %let it be constant for now
            sigma_squared = Sigma2;
            %inverse gamma
            %draw gamma
            gamma = binornd(1,1/2,[p 1]);
            %if mod(t,disp_int)==0 && plotting
            %    figure(1); plot(D01(1:t-1),'.'); title('$\|\gamma^{(1)} - \gamma^{(0)}\|_0$','interpreter','latex'); drawnow;
            %    figure(2); plot(MGAMMA1(1:t-1),'.'); title('$m^{(1)}_{\gamma}$','interpreter','latex'); drawnow;
            %    figure(3); histogram(LBF1(1:t-1),50); title('$\log(BF^{(1)})$','interpreter','latex'); drawnow;
            %    figure(4); plot(D12(1:t-1),'.'); title('$\|\gamma^{(1)} - \gamma^{(2)}\|_0$','interpreter','latex'); drawnow;
            %    mean(ACC1(1:t-1))
            %    mean(ACC2(1:t-1))
            %end
            
            %if t==5
            %    if strcmp(start_measure,'prior')
            %        gamma2 = binornd(1,q,[p 1]);
            %    end
            %    if strcmp(start_measure,'null')
            %        gamma2 = zeros(p,1);
            %    end
            %    mgamma2 = sum(gamma2);
            %end
            
            %u = unifrnd(0,1);
            %single = u<p_single;
            %if single
            %    j1 = ceil(p*rand());
            %    prop_gamma1 = gamma1;
            %    prop_gamma1(j1) = 1-prop_gamma1(j1);
            %    mgamma_prop1 = sum(prop_gamma1);
                
                % single flip: we can always choose to flip the same coordinate
                % ED: Is this supposed to be a coupling
            %    prop_gamma2 = gamma2;
            %    prop_gamma2(j1) = 1-prop_gamma2(j1);
            %    mgamma_prop2 = sum(prop_gamma2);
            %else
            %    mgamma_prop1 = mgamma1; mgamma_prop2 = mgamma2;
            %    Sgamma11 = find(gamma1==1);
            %    Sgamma01 = find(gamma1==0);
                
            %    Sgamma12 = find(gamma2==1);
            %    Sgamma02 = find(gamma2==0);
                
                
                % propose gamma1 in the usual way
            %    if ~isempty(Sgamma11) && ~isempty(Sgamma01)
            %        j1 = randsample(Sgamma11,1);
            %        k1 = randsample(Sgamma01,1);
            %        prop_gamma1 = gamma1;
            %        prop_gamma1(j1) = 0;
            %        prop_gamma1(k1) = 1;
            %    else
            %        prop_gamma1 = gamma1;
            %    end
                
            %    if ~isempty(Sgamma12) && ~isempty(Sgamma02)
                    % check the probability that we can make them agree
            %        pr_agree1 = sum(gamma1==1 & gamma2==1)/sum(gamma2==1);
            %        pr_agree0 = sum(gamma1==0 & gamma2==0)/sum(gamma2==0);
                    
%                     u1 = rand(); u0 = rand();
%                     if pr_agree1>u1
%                         j2 = j1;
%                     else
%                         j2 = randsample(setdiff(Sgamma12,Sgamma11),1);
%                     end
%                     if pr_agree0>u0
%                         k2 = k1;
%                     else
%                         k2 = randsample(setdiff(Sgamma02,Sgamma01),1);
%                     end
%                     prop_gamma2 = gamma2;
%                     prop_gamma2(j2) = 0;
%                     prop_gamma2(k2) = 1;
%                 else
%                     prop_gamma2 = gamma2;
%                 end
%                 
%             end
%             
%             v = unifrnd(0,1);
%             if ~all(prop_gamma1==gamma1)
%                 Mprop = logml(XX,Xy,yy,prop_gamma1,Tau2,n);
%                 Mcurr = logml(XX,Xy,yy,gamma1,Tau2,n);
%                 lBF1 = Mprop-Mcurr;
%                 if ~single
%                     pr = exp(lBF1);
%                 else
%                     if mgamma_prop1 > mgamma1
%                         pr = exp(log(mgamma1+1)-log(p-mgamma1)+lBF1);
%                     else
%                         pr = exp(log(p-mgamma1)-log(mgamma1+1)+lBF1);
%                     end
%                 end
%                 acc1 = pr>v;
%                 if acc1
%                     gamma1 = prop_gamma1;
%                     mgamma1 = sum(gamma1);
%                 end
%             end
%             
%             if ~all(prop_gamma2==gamma2)
%                 Mprop = logml(XX,Xy,yy,prop_gamma2,Tau2,n);
%                 Mcurr = logml(XX,Xy,yy,gamma2,Tau2,n);
%                 lBF2 = Mprop-Mcurr;
%                 if ~single
%                     pr = exp(lBF2);
%                 else
%                     if mgamma_prop2 > mgamma2
%                         pr = exp(log(mgamma2+1)-log(p-mgamma2)+lBF2);
%                     else
%                         pr = exp(log(p-mgamma2)-log(mgamma2+1)+lBF2);
%                     end
%                 end
%                 acc2 = pr>v;
%                 if acc2
%                     gamma2 = prop_gamma2;
%                     mgamma2 = sum(gamma2);
%                 end
%             end
%             
%             D01(t) = sum(abs(gamma1-GammaTrue));
%             TP1(t) = sum(1.*(gamma1==1 & GammaTrue==1));
%             FN1(t) = sum(1.*(gamma1==0 & GammaTrue==1));
%             MGAMMA1(t) = mgamma1;
%             LBF1(t) = lBF1;
%             
%             D02(t) = sum(abs(gamma2-GammaTrue));
%             TP2(t) = sum(1.*(gamma2==1 & GammaTrue==1));
%             FN2(t) = sum(1.*(gamma2==0 & GammaTrue==1));
%             
%             %GAMMA(t,:) = gamma;
%             ACC1(t) = acc1;
%             ACC2(t) = acc2;
%             
%             D12(t) = sum(abs(gamma2-gamma1));
%             
%         end
%         
%         
%         
%         if saving
%             save(strcat('Outputs/mh_bvs_',num2str(n),'_',num2str(p),'_',num2str(q*100),'_',num2str(r),'.mat'),'D01','D12','TP1','FN1','MGAMMA1','LBF1','ACC1');
         end
         
    end
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


