clf;
% automatically create postscript whenever
% figure is drawn
tmpfilename = 'figbaseline_v2';
tmpfilebwname = sprintf('%s_noname_bw',tmpfilename);
tmpfilenoname = sprintf('%s_noname',tmpfilename);

tmpprintname = fixunderbar(tmpfilename);
% for use with xfig and pstex
tmpxfigfilename = sprintf('x%s',tmpfilename);

tmppos= [0.2 0.2 0.7 0.7];
tmpa1 = axes('position',tmppos);

set(gcf,'DefaultLineMarkerSize',10);
% set(gcf,'DefaultLineMarkerEdgeColor','k');
% set(gcf,'DefaultLineMarkerFaceColor','w');
set(gcf,'DefaultAxesLineWidth',2);

set(gcf,'PaperPositionMode','auto');
set(gcf,'Position',[520 554 893 452]);


% main data goes here
% Good baseline parameters
pars.N=1;
pars.gamma=1/5;
pars.beta0=0.0;
pars.mu0=5*10^-3;
pars.R0=4;
pars.Dbeta1 = (pars.R0*pars.gamma-pars.beta0)*pars.mu0;
pars.I0=10^-6;
pars.betabar_exp = pars.beta0+pars.Dbeta1/pars.mu0;

options = odeset('RelTol',1e-7);
y0 = [1-pars.I0,pars.betabar_exp,pars.I0,0];

[t,y]=ode45(@sir_struct,[0:1:500],y0,options,pars);
[tmpa, tmph1,tmph2]=plotyy(t,log10(y(:,3)),t,y(:,2));
set(tmph1,'linewidth',3,'color','k');
set(tmph2,'linewidth',3,'color','r','linestyle',':');
set(tmpa(1),'ytick',[-6:1:0]);
set(tmpa(1),'ycolor','k');
tmpf=ylabel(tmpa(1),'Infected fraction, $I(t)$','verticalalignment','bottom','interpreter','latex','fontsize',20);
set(tmpa(2),'ycolor','r');
tmpf=ylabel(tmpa(2),'Avg. Susceptibility, $\bar{\beta}(t)$','interpreter','latex','fontsize',20);
xlabel('Days, $t$','verticalalignment','top','interpreter','latex','fontsize',20);
title('${\cal{R}}_0=4$, $\gamma=1/5$','interpreter','latex','fontsize',20);
axes(tmpa(1));
set(gca,'fontsize',20);
axes(tmpa(2));
set(gca,'fontsize',20);


% loglog(,, '');
%
%
% Some helpful plot commands
% tmph=plot(x,y,'ko');
% set(tmph,'markersize',10,'markerfacecolor,'k');
% tmph=plot(x,y,'k-');
% set(tmph,'linewidth',2);


% for use with layered plots
% set(gca,'box','off')

% adjust limits
% tmpv = axis;
% axis([]);
% ylim([]);
% xlim([]);

% change axis line width (default is 0.5)
% set(tmpa1,'linewidth',2)

% fix up tickmarks
% set(gca,'xtick',[1 100 10^4])
% set(gca,'ytick',[1 100 10^4])

% creation of postscript for papers
% psprint(tmpxfigfilename);

% the following will usually not be printed 
% in good copy for papers
% (except for legend without labels)

% legend
% tmplh = legend('stuff',...);
% tmplh = legend('','','');
% remove box
% set(tmplh,'visible','off')
% legend('boxoff');

% title('','fontsize',24)
% 'horizontalalignment','left');

% for writing over the top
% coordinates are normalized again to (0,1.0)
tmpa2 = axes('Position', tmppos);
set(tmpa2,'visible','off');
% first two points are normalized x, y positions
% text(,,'','Fontsize',14);

% automatic creation of postscript
% without name/date
psprintc(tmpfilenoname);
psprint(tmpfilebwname);

tmpt = pwd;
tmpnamememo = sprintf('[source=%s/%s.ps]',tmpt,tmpprintname);
text(1.05,.05,tmpnamememo,'Fontsize',6,'rotation',90);
datenamer(1.1,.05,90);
% datename(.5,.05);
% datename2(.5,.05); % 2 rows

% automatic creation of postscript
psprintc(tmpfilename);

% set following on if zooming of 
% plots is required
% may need to get legend up as well
%axes(tmpa1)
%axes(tmplh)
clear tmp*
