% Compare PF with Kalman filter
rng('default')
clear;
close all;

numSamples=20;        % Number of Monte Carlo samples per time step. 


T = 10;                % Number of time steps.
t = 1:1:T;             % Time.  
x0 = 0.0;              % Initial state. UNIFORM(0,1)
true_x = zeros(T,1);        % Hidden states.
y = zeros(T,1);        % Observations.
true_x(1,1) = x0;           % Initial state. 
R =    2;                 % Measurement noise variance.
Q =    1;                % Process noise variance.     

% just for plotting stuff
grid=linspace(-8,8,100);


% LQG model parameters
a=0.9;
b=sqrt(Q);
c=.5;
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
q(:,1) = qq/sum(qq);

I(:,1)=randsample(1:N,N,true,q(:,1));
x(:,1)= xu(I(:,1),1);

% % uncomment to save a video
% v = VideoWriter('pf.mp4');
% open(v);
figure(13)
hold on
xlabel('time $n$', 'Interpreter','latex')
ylabel('$X^i_n$', 'Interpreter','latex')
for time=1:T
    plot(time*ones(size(grid)),grid,'b')
end
plot(ones(N,1),xu(:,1),'ko')

viewfactor=2;
likeli=exp(-.5*(R^(-1))*(y(1)- grid).^(2))/(sqrt(2*pi*R));
plot((1)*ones(size(grid))+viewfactor*likeli,grid,'r')

pause

p1=plot(ones(N,1),x(:,1),'ro');

pause


% % uncomment to save a video
% frame = getframe(gcf);
% writeVideo(v,frame);


for t=1:T-1
    
    w = sqrt(Q)*randn(N,1);
    xu(:,t+1) = a*x(:,t)+w;
    
    qq = zeros(N,1);% unnormalised weights
    m = c*xu(:,t+1);
    for s=1:N
        qq(s,t+1) = exp(-.5*(R^(-1))*(y(t+1)- m(s,1))^(2))/(2*pi*sqrt(R));
    end
    q(:,t+1) = qq(:,t+1)/sum(qq(:,t+1));
    %  sampler offsprings
    I(:,t+1)=randsample(1:N,N,true,q(:,t+1));
    % resampling     
    x(:,t+1)= xu(I(:,t+1),t+1);
    x(:,1:t)= x(I(:,t+1),1:t);
    
    %   % Plotting particle system     
    
    % % uncomment to save a video
    %     frame = getframe(gcf);
    %    writeVideo(v,frame);
        
    
    Xpoint=[(t)*ones(N,1),(t+1)*ones(N,1)];
    Ypoint=[xu(I(:,t),t),xu(:,t+1)];
    
    
    
    line(Xpoint',Ypoint','Color','cyan','LineStyle','--')
    % % uncomment to save a video
    %     frame = getframe(gcf);
    %    writeVideo(v,frame);
    plot((t+1)*ones(size(xu(:,t+1))),xu(:,t+1),'ko')

    pause
    
    
    likeli=exp(-.5*(R^(-1))*(y(t+1)- grid).^(2))/(2*pi*sqrt(R));
    plot((t+1)*ones(size(grid))+viewfactor*likeli,grid,'r')
    
    
    pause    
    
    plot((t+1)*ones(N,1),xu(I(:,t+1),t+1),'ro');

    pause
    
        set(p1,'Visible','off')
        p1=plot(x(:,1:t+1)','k');
        set(p1,'Visible','on')
        % % uncomment to save a video
        %         frame = getframe(gcf);
        %         writeVideo(v,frame);
        pause
    
    
end


% % uncomment to save a video
% close(v);


% % calculations for likelihood
% rec=(mean(qq,1));
% rec_true=exp(-.5*(RecLikeVar.^(-1)).*(y-RecLikeMean).^2)./(2*pi*sqrt(RecLikeVar));








