% Compare SIS with Kalman filter
rng('default')
clear;
close all
tic;
numSamples=500;        % Number of Monte Carlo samples per time step. 


T = 100;                % Number of time steps.
t = 1:1:T;             % Time.  
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


% recursive likelihood
RecLikeMean=zeros(T,1);
RecLikeVar=RecLikeMean;

% KF predictor and update means and variance
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



rec_true=exp(-.5*(RecLikeVar.^(-1)).*(y-RecLikeMean).^2)./(sqrt(2*pi*RecLikeVar));...
    %normpdf(y,RecLikeMean,sqrt(RecLikeVar));



% bootstrap SIS

N = numSamples;             % Number of particles;

% filtering approx
x=zeros(N,T);
% prediction approx
xu=zeros(N,T);
% normalised weights
q=zeros(N,T);
% unormalised weights
qq=q;



% INIT: SAMPLE FROM THE PRIOR:
x(:,1) = x0+sqrt(initVar)*randn(N,1);

qq = zeros(N,1);% unnormalised weights
m = c*x(:,1);
for s=1:N    
    qq(s,1) = exp(-.5*R^(-1)*(y(1)- m(s,1))^(2))/(sqrt(2*pi*R));  
end
q(:,1) = qq(:,1)/sum(qq(:,1));


% UPDATE AND PREDICTION STAGES:

for t=1:T-1
    
    w = sqrt(Q)*randn(N,1);
    x(:,t+1) = a*x(:,t)+w;
    
    m = c*x(:,t+1);
    for s=1:N
        qq(s,t+1) = qq(s,t)*exp(-.5*(R^(-1))*(y(t+1)- m(s,1))^(2))/(sqrt(2*pi*R));
    end
    q(:,t+1) = qq(:,t+1)/sum(qq(:,t+1));
end


    
simtime=toc
figure(1)
% xlabel('time n')
% ylabel('$E[X_n]$','Interpreter' ,'latex')
plot(mu)
hold on
plot(sum(q.*x))
figure(2)
% xlabel('time n')    
% ylabel('$E[X_n^2]-E[X_n]^2$','Interpreter' ,'latex')
plot(Sigma)
hold on
plot(sum(q.*(x.*x))-(sum(q.*x).^2))
figure(3)
xlabel('time n')
ylabel('$\log p(y_{0:n})$','Interpreter' ,'latex')
hold on
rec=(mean(qq,1));
plot((log(rec)))

plot(cumsum(log(rec_true)),'-r')

figure(4)
hold on
xlabel('time n')
ylabel('$ESS_n$','Interpreter' ,'latex')
plot(sum(q.*q).^-1)
[B,ind]=sort(q(:,T));
figure(5)
xlabel('time n')
ylabel('$X^i_{0:T}$','Interpreter' ,'latex')
title('Samples using bootstrap SIS - color code based on weight')
hold on
colormap(autumn)
for ll=ind'
    patch(1:T+1,[x(ll,:),NaN],-[q(ll,:),NaN],'EdgeColor','interp','MarkerFaceColor','flat');
end
% colorbar
plot(mu,'-.b')

% % optimal proposal with SIS

% Optimal SIS

% filtering approx
x=zeros(N,T);
% normalised weights
q=zeros(N,T);
% unormalised weights
qq=q;

% Opt proposal statistics

S=1/(1/Q+c*c/R);
mu_prop_offset=(c/R*y)*S;
mu_prop_gain=Q*a*S;

% INIT: SAMPLE FROM THE PRIOR:
x(:,1) = x0+sqrt(initVar)*randn(N,1);

qq = zeros(N,T);% unnormalised weights
m = c*x(:,1);
for s=1:N  
    qq(s,1) = exp(-.5*R^(-1)*(y(1)- m(s,1))^(2))/(sqrt(2*pi*R));   
end
q(:,1) = qq(:,1)/sum(qq(:,1));

% UPDATE AND PREDICTION STAGES:

for t=1:T-1
    
    x(:,t+1) = mu_prop_offset(t+1)*ones(N,1)+mu_prop_gain*x(:,t)+sqrt(S)*randn(N,1);
    
    
     for s=1:N
                qq(s,t+1) = qq(s,t)*exp(-.5*(R^(-1))*(y(t+1)- c*x(s,t+1))^(2))/(sqrt(2*pi*R))...
                    *exp(-0.5*(Q^-1)*( x(s,t+1) - a*x(s,t))^2)/(sqrt(2*pi*Q))/...
                    exp(-0.5*(S^-1)*(x(s,t+1) - mu_prop_offset(t+1)-mu_prop_gain*x(s,t))^2)*...
                    (sqrt(2*pi*S));
                
%                 qq(s,t+1) = qq(s,t)*exp(-y(t+1)^2/R/2-(a*x(s,t))^2/2/Q)...
%                     /(sqrt(2*pi*Q))/(sqrt(2*pi*R))...
%                     *sqrt(2*pi/(S^-1))*exp((c*y(t+1)/R+a*x(s,t)/Q)/2/(S^-1));
                
    end
    % implementing log exp sum trick for weights
    q(:,t+1) = qq(:,t+1)/sum(qq(:,t+1));
           
end

simtime=toc

figure(1)
plot(sum(q.*x))
legend('KF','BSIS','OptimalSIS')
xlabel('time n')    
ylabel('$E[X_n|Y_{0:n}]$','Interpreter' ,'latex')
figure(2)
plot(sum(q.*(x.*x))-(sum(q.*x).^2))
legend('KF','BSIS','OptimalSIS')
xlabel('time n')    
ylabel('$E[X_n^2|Y_{0:n}]-E[X_n|Y_{0:n}]^2$','Interpreter' ,'latex')
figure(3)
rec=(mean(qq,1));
plot((log(rec)))
legend('KF','BSIS','OptimalSIS')
figure(4)
plot(sum(q.*q).^-1)
legend('BSIS','OptimalSIS')
figure(6)
[B,ind]=sort(q(:,T));
hold on
xlabel('time n')
ylabel('$X^i_{0:T}$','Interpreter' ,'latex')
title('Samples using optimal SIS - color code based on weight')
hold on
colormap(autumn)
for ll=ind'
    patch(1:T+1,[x(ll,:),NaN],-[q(ll,:),NaN],'EdgeColor','interp','MarkerFaceColor','flat');
end
% colorbar
plot(mu,'-.b')
clear