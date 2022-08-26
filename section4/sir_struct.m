function dydt = sir_struct(t,y,pars)
% function dydt = sir_struct(t,y,pars)
%
% Follows the Social Diffusion + Hetero mean field equations

dydt=zeros(4,1);
S=y(1);
betabar=y(2);
I=y(3);
R=y(4);

dydt(1) = -betabar*S*I/pars.N;
dydt(2) = -2*I/pars.N*(betabar-pars.beta0)^2-2*pars.mu0*(betabar-pars.beta0)+2*pars.Dbeta1;
dydt(3) = betabar*S*I/pars.N-pars.gamma*I;
dydt(4) = pars.gamma*I;
