% Compare PF with Kalman filter
rng('default')
clear;
close all;
tic;
numSamples=500;        % Number of Monte Carlo samples per time step. 


T = 100;                % Number of time steps.
x0 = 0.0;              % Initial state. UNIFORM(0,1)
true_x = zeros(T,1);        % Hidden states.
y = zeros(T,1);        % Observations.
true_x(1,1) = x0;           % Initial state. 

R =    2;                 % Measurement noise variance.
Q =    1;                % Process noise variance.      

% LQG model parameters
a=0.6;
b=sqrt(Q);
c=1;
d=sqrt(R);

initVar = 4;           % Initial variance of the states.  

v = randn(T,1);
w = randn(T,1);


% GENERATE TRUE STATE AND MEASUREMENTS:

y(1,1) = c*true_x(1,1) + d*v(1,1); 
for t=2:T
  true_x(t,1) = a*true_x(t-1,1) + b*w(t,1);
  y(t,1) = c*true_x(t,1) + d*v(t,1); 
end


% Kalman Filter

RecLikeMean=zeros(T,1);
RecLikeVar=RecLikeMean;
mu=zeros(T,1);
Sigma=zeros(T,1);
mu(1,1)=x0;
Sigma(1,1)=initVar;


RecLikeMean(1,1)=x0;
RecLikeVar(1,1)=c*initVar*c'+d*d';
    
for t=2:T
    
    mu_pred=a*mu(t-1,:);
    SigmaPred=a*Sigma(t-1,:)*a'+b*b';
    
    z=y(t,1)-c*mu_pred;
    SS=c*SigmaPred*c'+d*d';
    
    RecLikeMean(t,1)=mu_pred;
    RecLikeVar(t,1)=SS;
    
    K=SigmaPred*c'*inv(SS);    
    Sigma(t,1)=(1-K*c)*SigmaPred;
    mu(t,1)=mu_pred+K*z;
    
end


% bootstrap PF

N = numSamples;             % Number of particles;

% filtering approx
x=zeros(N,T);
% sampled particles
xu=zeros(N,T);
% normalised weights
q=zeros(N,T);
% unormalised weights
qq=q;


% ancestors index
I=zeros(size(x));


% INIT: SAMPLE FROM THE PRIOR:
xu(:,1) = x0+sqrt(initVar)*randn(N,1);

qq = zeros(N,1);% unnormalised weights
m = c*xu(:,1);
for s=1:N    
    qq(s,1) = exp(-.5*R^(-1)*(y(1)- m(s,1))^(2))/(sqrt(2*pi*R));    
end
q(:,1) = qq(:,1)/sum(qq(:,1));

I(:,1)=randsample(1:N,N,true,q(:,1));
x(:,1)= xu(I(:,1),1);
% UPDATE AND PREDICTION STAGES:

for t=1:T-1
    
    w = sqrt(Q)*randn(N,1);
    xu(:,t+1) = a*x(:,t)+w;
    
    m = c*xu(:,t+1);
    for s=1:N
        qq(s,t+1) = exp(-.5*(R^(-1))*(y(t+1)- m(s,1))^(2))/(sqrt(2*pi*R));
    end
    q(:,t+1) = qq(:,t+1)/sum(qq(:,t+1));
    %  sampler offsprings
    I(:,t+1)=randsample(1:N,N,true,q(:,t+1));
    % resampling     
    x(:,t+1)= xu(I(:,t+1),t+1);
    x(:,1:t)= x(I(:,t+1),1:t);
end


rec=(mean(qq,1));

rec_true=exp(-.5*(RecLikeVar.^(-1)).*(y-RecLikeMean).^2)./(sqrt(2*pi*RecLikeVar));
    %normpdf(y,RecLikeMean,sqrt(RecLikeVar));
    
simtime=toc

figure(1)
% xlabel('time n')
% ylabel('$E [ X_n ]$','Interpreter' ,'latex')
plot(mu)
hold on
plot(sum(q.*xu))
figure(2)
% xlabel('time n')
% ylabel('$E[ X_n^2 ]-E[ X_n ]^2$','Interpreter' ,'latex')
plot(Sigma)
hold on
plot(sum(q.*(xu.*xu))-(sum(q.*xu).^2))
figure(3)
xlabel('time n')
ylabel('$\log p(y_{0:n})$','Interpreter' ,'latex')
hold on
plot(cumsum(log(rec)))
plot(cumsum(log(rec_true)),'-r')
figure(4)
xlabel('time n')
ylabel('$ESS_n$','Interpreter' ,'latex')
hold on
plot(sum(q.*q).^-1)

figure(5)
hold on
xlabel('time n')
ylabel('$X^i_{0:T}$','Interpreter' ,'latex')
title('Resampled particles at final time - BPF')
plot(x','k')
plot(mu,'-.r')

% Optimal PF

% filtering approx
x=zeros(N,T);
% sampled particles
xu=zeros(N,T);
% normalised weights
q=zeros(N,T);
% unormalised weights
qq=q;
log_w=q;
% ancestors index
I=zeros(size(x));

log_rec=zeros(size(rec));

% Opt proposal statistics
S=((b*b)^-1+c*((d^2)^-1)*c)^-1;
mu_prop_offset=(S)*(c*((d*d)^-1)*y);
mu_prop_gain=(S)*((b*b)^-1)*a;

% INIT: SAMPLE FROM THE PRIOR:
xu(:,1) = x0+sqrt(initVar)*randn(N,1);

qq = zeros(N,T);% unnormalised weights
m = c*xu(:,1);
for s=1:N  
    qq(s,1) = exp(-.5*R^(-1)*(y(1)- m(s,1))^(2))/(sqrt(2*pi*R));   
end
q(:,1) = qq(:,1)/sum(qq(:,1));
I(:,1)=randsample(1:N,N,true,q(:,1));
x(:,1)= xu(I(:,1),1);
log_rec(:,1)=log(mean(qq(:,1)));

% UPDATE AND PREDICTION STAGES:

for t=1:T-1
    
    xu(:,t+1) = mu_prop_offset(t+1)+mu_prop_gain*x(:,t)+sqrt(S)*randn(N,1);
    
    
    for s=1:N
        %         qq(s,t+1) = exp(-.5*(R^(-1))*(y(t+1)-c*x(:,t+1))^(2))/(sqrt(2*pi*R))...
        %             *exp(-0.5*(Q^-1)*( xu(s,t+1) - a*x(s,t))^2)/(sqrt(2*pi*Q))/...
        %             exp(-0.5*(S^-1)*(xu(s,t+1) - mu_prop_offset(t+1)-mu_prop_gain*x(s,t))^2)*...
        %             (sqrt(2*pi*S));
        
        log_w(s,t+1) = -0.5*(R^(-1))*(y(t+1)- c*xu(s,t+1))^2 ...
            -0.5*(Q^-1)*( xu(s,t+1) - a*x(s,t))^2 ...
            +0.5*(S^-1)*(xu(s,t+1) - mu_prop_offset(t+1)-mu_prop_gain*x(s,t))^2 ...
            -log(sqrt(2*pi*R))-log(sqrt(2*pi*Q))+log(sqrt(2*pi*S));
        
    end
    
    % implementing log exp sum trick for weights
    offset=max(log_w(:,t+1));
    log_w(:,t+1)=log_w(:,t+1)-offset;
    
    qq(:,t+1)=exp(log_w(:,t+1));
    q(:,t+1) = qq(:,t+1)/sum(qq(:,t+1));
    
    log_rec(:,t+1)=log(mean(qq(:,t+1)))+offset;
    
    %  sampler offsprings
    I(:,t+1)=randsample(1:N,N,true,q(:,t+1));
    % resampling
    x(:,t+1)= xu(I(:,t+1),t+1);
    x(:,1:t)= x(I(:,t+1),1:t);
end

simtime=toc

figure(1)
plot(sum(q.*xu))
legend('KF','BPF','OptimalPF')
xlabel('time n')
ylabel('$E[ X_n |Y_{0:n}]$','Interpreter' ,'latex')
figure(2)
plot(sum(q.*(xu.*xu))-(sum(q.*xu).^2))
legend('KF','BPF','OptimalPF')
xlabel('time n')
ylabel('$E[ X_n^2 |Y_{0:n} ]-E[ X_n |Y_{0:n} ]^2$','Interpreter' ,'latex')
figure(3)
plot(cumsum(log_rec))
legend('KF','BPF','OptimalPF')
figure(4)
plot(sum(q.*q).^-1)
legend('BPF','OptimalPF')
figure(6)
hold on
xlabel('time n')
ylabel('$X^i_{0:T}$','Interpreter' ,'latex')
title('Resampled particles at final time - Optimal PF')
plot(x','k')
plot(mu,'-.r')




clear