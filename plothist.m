numpoints=1000;

load('mcmc_run_temp_vlow.mat')
ksdensity(SIGMA2,'npoints',numpoints)
hold on
plot(0:0.1:7,inverse_gamma_unnormalised2(0:0.1:7,a0+dim_y/2,b0+sum(y.^2)/2),'k-.')
xlim([0,5])
saveas(gcf,'hist1.eps')
close

load('mcmc_run_temp_low.mat')
ksdensity(SIGMA2,'npoints',numpoints)
hold on
plot(0:0.1:7,inverse_gamma_unnormalised2(0:0.1:7,a0+dim_y/2,b0+sum(y.^2)/2),'k-.')
xlim([0,5])
saveas(gcf,'hist2.eps')
close

load('mcmc_run_temp.mat')
ksdensity(SIGMA2,'npoints',numpoints)
hold on
plot(0:0.1:7,inverse_gamma_unnormalised2(0:0.1:7,a0+dim_y/2,b0+sum(y.^2)/2),'k-.')
xlim([0,5])
saveas(gcf,'hist3.eps')
close

load('mcmc_run_temp_high.mat')
ksdensity(SIGMA2,'npoints',numpoints)
hold on
plot(0:0.1:7,inverse_gamma_unnormalised2(0:0.1:7,a0+dim_y/2,b0+sum(y.^2)/2),'k-.')
xlim([0,5])
saveas(gcf,'hist4.eps')
close

load('mcmc_run_temp_vhigh.mat')
ksdensity(SIGMA2,'npoints',numpoints)
hold on
plot(0:0.1:7,inverse_gamma_unnormalised2(0:0.1:7,a0+dim_y/2,b0+sum(y.^2)/2),'k-.')
xlim([0,5])
saveas(gcf,'hist5.eps')
close
