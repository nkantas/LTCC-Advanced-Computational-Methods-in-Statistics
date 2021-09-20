

clear

rand('twister', sum(100*clock));
randn('state',sum(100*clock));



% Declare likelihood and prior parameters
m0=0; C0=1; % x0 is N(m0,C0)
a0=1; b0=1; % sigma2 is IG(a0,b0)

dim_y=5;
y=randn(dim_y,1)*sqrt(C0);

% NOTE on inline for simplicity assume a0=c0 and b0=d0
inverse_gamma_unnormalised=inline('(x^(-a-1))*exp(-(b/x))','x','a','b');
inverse_gamma_unnormalised2=inline('(b^a)*(x.^(-a-1)).*exp(-(b./x))/gamma(a)','x','a','b');
norm_log_pdf=inline('-0.5*y.^2/sigma2-ones(size(y))*log(sqrt(2*pi*sigma2))','y','sigma2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% IMPUTE FIRST SAMPLE by sampling from the prior

sigma2Sample =1./ gamrnd(a0,1/b0,1,1);


% product of un-normalised priors
Prior=inverse_gamma_unnormalised(sigma2Sample,a0,b0);

% log posterior
logYsum=sum(norm_log_pdf(y,sigma2Sample))+log(Prior);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Start Metropolis Hastings

rep=100000;

SIGMA2=zeros(rep,1);
LOGY=SIGMA2;
RATIO=[];
tic
ll=1;
while ll<=rep

    % Random Walk move with Gaussian
    % Set initial condition to true guy   xSampleNew=xSample+sqrt(0.05)*randn(1,1);

    rw_step=10;

    sigma2SampleNew = sigma2Sample+rw_step*randn(1,1);

    while sigma2SampleNew <=0
        sigma2SampleNew = sigma2Sample+rw_step*randn(1,1);
    end

    % New product of un-normalised priors
    PriorNew=inverse_gamma_unnormalised(sigma2SampleNew,a0,b0);

    % % % % % % % % % % % % % %
    logYsumNew=sum(norm_log_pdf(y,sigma2SampleNew))+log(PriorNew);
    % % % % % % % % % %

    ratio=exp(logYsumNew-logYsum);

    accept_ratio=min(1,ratio);

    RATIO=[RATIO accept_ratio];

    u=rand(1);
    if u<=accept_ratio && ~isnan(ratio) && logYsumNew~=0
        SIGMA2(ll,1)=sigma2SampleNew;
       
        logYsum=logYsumNew;

        sigma2Sample=sigma2SampleNew;
       
    else

        SIGMA2(ll,1)=sigma2Sample;
      

    end

    ll=ll+1;
    if mod(ll,10000)==0
        disp(['completed iteration ' num2str(ll) ' .... after ' num2str(size(RATIO,2)) ' attempts .......'])
        save mcmc_run_temp.mat
    end

end

simtime=toc





