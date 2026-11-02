function render_adopted_result_figures(onlyFigure11,onlyFigure6)
% Approved paper figures; archived native FIGs remain read-only inputs.
% Visual inspiration: NeurIPS 2021 rliable interval estimates (style only).
% Retain raw units, all five metrics, every condition, and frozen confidence limits.
% Pass true for Figure 11 only, preserving the Figure 6 exports byte-for-byte.
% Pass false,true for Figure 6 only, preserving the Figure 11 exports.
if nargin<1,onlyFigure11=false;end
if nargin<2,onlyFigure6=false;end
assert(~(onlyFigure11 && onlyFigure6));
here=fileparts(mfilename('fullpath'));root=fileparts(here);out=fullfile(root,'figures');
qadir=fullfile(root,'figures','render_checks');if ~isfolder(qadir),mkdir(qadir);end
assert(any(strcmp(listfonts,'Cambria Math')));
T=readtable(fullfile(root,'supplementary','results','plot_ci.csv'),'TextType','string');
names=["Qwen-VL-Max","Qwen-VL-Plus","Qwen3-VL-Flash","Qwen3.5-35B-A3B","Qwen3.5-Flash"];
metric=["SR","NE","SPL","OR","TL"];
values=zeros(5,5);lo=values;hi=values;
for m=1:5
 for j=1:5
  r=T(T.environment=="City Park" & T.condition==names(j) & T.metric==metric(m),:);
  assert(height(r)==1);values(m,j)=r.mean;lo(m,j)=r.low;hi(m,j)=r.high;
 end
end
% Check against the published native FIG, which remains read-only.
source=openfig(fullfile(root,'figures','map1_base_models.fig'),'invisible');
labels={'SR (%)','NE (m)','SPL (%)','OR (%)','TL (m)'};
for m=1:5
 a=axis_for(source,labels{m});b=findall(a,'Type','bar');
 assert(numel(b)==1 && max(abs(b.YData-values(m,:)))<1e-9);
end
close(source);
colors=[.47 .52 .58;.53 .43 .66;.33 .60 .71;.08 .47 .49;.85 .48 .24];
f=newfig([18.3 6.5]);
% Context and uncertainty definitions are in the manuscript captions.
left=.235;gap=.027;w=.126;bottom=.12;h=.80;
limits=[0 70;0 140;0 70;0 105;0 185];
titles={'SR (%)  ↑','NE (m)  ↓','SPL (%)  ↑','OR (%)  ↑','TL (m)'};
for m=1:5
 a=axes(f,'Position',[left+(m-1)*(w+gap) bottom w h]);hold(a,'on');
 for j=1:5
  y=6-j;
  plot(a,[lo(m,j) hi(m,j)],[y y],'-','Color',colors(j,:),'LineWidth',1.1,'Tag','FrozenCI');
  for bound=[lo(m,j) hi(m,j)]
   plot(a,[bound bound],y+[-.045 .045],'-','Color',colors(j,:),'LineWidth',1.1,'Tag','Figure6IntervalCap');
  end
  plot(a,values(m,j),y,'o','MarkerSize',4.4,'MarkerFaceColor',colors(j,:), ...
      'MarkerEdgeColor','w','LineWidth',.5,'Tag','FrozenMean');
  text(a,values(m,j),y+.20,sprintf('%.1f',values(m,j)), ...
      'FontName','Cambria Math','FontSize',7.1,'HorizontalAlignment','center', ...
      'VerticalAlignment','bottom','Color',[.18 .21 .25],'Tag','ValueLabel');
  if m==1
   text(a,-.12*diff(limits(m,:)),y,char(names(j)), ...
      'FontName','Cambria Math','FontSize',8,'HorizontalAlignment','right', ...
      'VerticalAlignment','middle','Clipping','off','Color',colors(j,:), ...
      'FontWeight','bold');
  end
 end
 xlim(a,limits(m,:));ylim(a,[.55 5.85]);
 styleaxis(a);a.YTick=[];a.YColor='none';a.XGrid='on';
 title(a,titles{m},'FontName','Cambria Math','FontSize',8.5,'FontWeight','bold');
end
if ~onlyFigure11,export_all(f,out,'map1_backbone_intervals');end
close(f);

H=readtable(fullfile(root,'source_data','heading_ablation_summary.csv'),'TextType','string');
D=readtable(fullfile(root,'source_data','heading_ablation_difference_ci.csv'),'TextType','string');
groups=["map1_ablated","map1_full","map2_ablated","map2_full"];
v=zeros(5,4);l=v;u=v;
for m=1:5
 for j=1:4
  r=H(H.group==groups(j) & H.metric==metric(m),:);assert(height(r)==1);
  v(m,j)=r.mean;l(m,j)=r.ci_low;u(m,j)=r.ci_high;
 end
end
source=openfig(fullfile(root,'figures','heading_ablation_comparison.fig'),'invisible');
for m=1:5
 a=axis_for(source,labels{m});b=findall(a,'Type','bar');
 assert(numel(b)==1 && max(abs(b.YData-v(m,:)))<1e-9);
end
close(source);
mapColors=[.08 .47 .49;.56 .37 .65];
f=newfig([18.3 9.4]);
% Categorical color bars identify maps; marker fill identifies the ablation.
mapLegendAx=axes(f,'Position',[.015 .93 .39 .055]);hold(mapLegendAx,'on');
hm1=plot(mapLegendAx,NaN,NaN,'-','Color',mapColors(1,:),'LineWidth',3,'Tag','MapColorBar');
hm2=plot(mapLegendAx,NaN,NaN,'-','Color',mapColors(2,:),'LineWidth',3,'Tag','MapColorBar');
legend(mapLegendAx,[hm1 hm2],{'City Park','Downtown West'},'NumColumns',2,'Box','off', ...
 'FontName','Cambria Math','FontSize',8.2,'Location','west');axis(mapLegendAx,'off');
legendAx=axes(f,'Position',[.46 .93 .53 .055]);hold(legendAx,'on');
ha=plot(legendAx,NaN,NaN,'o','MarkerFaceColor','w','MarkerEdgeColor',[.25 .28 .33],'MarkerSize',5);
hu=plot(legendAx,NaN,NaN,'o','MarkerFaceColor',[.25 .28 .33],'MarkerEdgeColor',[.25 .28 .33],'MarkerSize',5);
legend(legendAx,[ha hu],{'Heading Prediction ablated','Unablated'},'NumColumns',2,'Box','off', ...
 'FontName','Cambria Math','FontSize',8.2,'Location','west');axis(legendAx,'off');

% Primary outcome in its original scale.
a=axes(f,'Position',[.095 .44 .34 .40]);hold(a,'on');
for p=1:2
 ids=(2*p-1):(2*p);c=mapColors(p,:);
 xs=[1 2]+.035*(2*p-3);
 plot(a,xs,v(1,ids),'--','Color',[.58 .61 .64],'LineWidth',.85,'Tag','ConditionConnector');
 for j=1:2
  x=j+(.035*(2*p-3));k=ids(j);
  errorbar(a,x,v(1,k),v(1,k)-l(1,k),u(1,k)-v(1,k),'Color',c, ...
   'LineStyle','none','LineWidth',1.1,'CapSize',4,'Tag','FrozenMarginalCI');
  face='w';if j==2,face=c;end
  plot(a,x,v(1,k),'o','Color',c,'MarkerFaceColor',face,'MarkerSize',5,'Tag','FrozenMean');
  if j==1
   labelX=x-.11;labelY=v(1,k);align='right';
   if p==2,labelX=x+.14;labelY=labelY-4;align='left';end
   text(a,labelX,labelY,sprintf('%.1f',v(1,k)),'Color',c,'HorizontalAlignment',align, ...
    'VerticalAlignment','middle','FontName','Cambria Math','FontSize',8.5,'FontWeight','bold');
  else
   yy=u(1,k)+4;va='bottom';if p==2,yy=l(1,k)-5;va='top';end
   text(a,x,yy,sprintf('%.1f',v(1,k)),'Color',c,'HorizontalAlignment','center', ...
    'VerticalAlignment',va,'FontName','Cambria Math','FontSize',8.5,'FontWeight','bold');
  end
 end
end
xlim(a,[.55 2.45]);ylim(a,[-5 103]);styleaxis(a);a.YTick=0:25:100;
a.XTick=[1 2];a.XTickLabel={'Ablated','Unablated'};a.YGrid='on';
title(a,'(a) SR (%)','FontSize',9,'FontWeight','bold');

% Paired primary contrast, never inferred from marginal whisker overlap.
b=axes(f,'Position',[.62 .475 .32 .32]);hold(b,'on');deltaSR=zeros(2,3);
for p=1:2
 map="map"+p;r=D(D.map==map & D.metric=="SR",:);assert(height(r)==1);
 deltaSR(p,:)=[r.mean r.ci_low r.ci_high];y=3-p;c=mapColors(p,:);
 plot(b,[r.ci_low r.ci_high],[y y],'-','Color',c,'LineWidth',1.1,'Tag','FrozenPairedCI');
 for bound=[r.ci_low r.ci_high]
  plot(b,[bound bound],y+[-.045 .045],'-','Color',c,'LineWidth',1.1,'Tag','IntervalCap');
 end
 plot(b,r.mean,y,'d','MarkerFaceColor',c,'MarkerEdgeColor',c,'MarkerSize',5,'Tag','FrozenPairedMean');
 text(b,r.mean,y+.25,sprintf('+%.1f [%.1f, %.1f]',r.mean,r.ci_low,r.ci_high), ...
  'HorizontalAlignment','center','FontSize',8.3,'FontWeight','bold','Color',c,'FontName','Cambria Math');
end
xlim(b,[0 100]);ylim(b,[.5 2.7]);styleaxis(b);b.YTick=[1 2];b.YTickLabel={'Downtown West','City Park'};
b.YColor=[.28 .31 .35];b.XGrid='on';
xlabel(b,'Unablated - Ablated','FontSize',8);
title(b,'(b) ΔSR (pp)','FontSize',9,'FontWeight','bold');

% Remaining four metrics use compact condition contrasts in raw units.
order=[3 2 4 5];shortTitles={'(c) SPL (%)  ↑','(d) NE (m)  ↓','(e) OR (%)  ↑','(f) TL (m)'};
lims=[0 90;0 60;0 100;0 65];
for t=1:4
 m=order(t);a=axes(f,'Position',[.075+(t-1)*.24 .10 .19 .23]);hold(a,'on');
 for p=1:2
  ids=(2*p-1):(2*p);c=mapColors(p,:);y=3-p;
  plot(a,v(m,ids),[y+.14 y-.14],'--','Color',[.58 .61 .64],'LineWidth',.85,'Tag','ConditionConnector');
  for j=1:2
   k=ids(j);yy=y+.14*(3-2*j);
   plot(a,[l(m,k) u(m,k)],[yy yy],'-','Color',c,'LineWidth',1.1,'Tag','FrozenMarginalCI');
   for bound=[l(m,k) u(m,k)]
    plot(a,[bound bound],yy+[-.035 .035],'-','Color',c,'LineWidth',1.1,'Tag','IntervalCap');
   end
   face='w';if j==2,face=c;end
   plot(a,v(m,k),yy,'o','MarkerFaceColor',face,'MarkerEdgeColor',c,'MarkerSize',4.4,'LineWidth',.8,'Tag','FrozenMean');
  end
  text(a,mean(lims(t,:)),y+.34,sprintf('%.1f  %s  %.1f',v(m,ids(1)),char(8594),v(m,ids(2))), ...
   'HorizontalAlignment','center','FontName','Cambria Math','FontSize',7.6,'Color',c,'Interpreter','none');
 end
 xlim(a,lims(t,:));ylim(a,[.60 2.75]);styleaxis(a);a.YTick=[];a.YColor='none';a.XGrid='on';
 if t==1
  % Shared map-row labels for the aligned lower panels; all plot positions stay fixed.
  text(a,-.04*diff(lims(t,:)),2,'City Park','FontName','Cambria Math','FontSize',7.3, ...
   'HorizontalAlignment','right','VerticalAlignment','middle','Color',[.28 .31 .35], ...
   'Clipping','off','Tag','MapRowLabel');
  text(a,-.04*diff(lims(t,:)),1,{'Downtown';'West'},'FontName','Cambria Math','FontSize',7.3, ...
   'HorizontalAlignment','right','VerticalAlignment','middle','Color',[.28 .31 .35], ...
   'Clipping','off','Tag','MapRowLabel');
 end
 title(a,shortTitles{t},'FontSize',8.4,'FontWeight','bold');
end
if ~onlyFigure6,export_all(f,out,'heading_ablation_intervals');end
close(f);
record=struct('passed',true,'backend','MATLAB','version',version,'no_statistics_recomputed',true, ...
 'figure6',struct('means',values,'low',lo,'high',hi,'models',names,'metric_order',metric), ...
 'figure11',struct('means',v,'low',l,'high',u,'group_order',groups,'metric_order',metric,'paired_sr',deltaSR));
fid=fopen(fullfile(qadir,'matlab_data_verification.json'),'w');fwrite(fid,jsonencode(record,PrettyPrint=true));fclose(fid);
fprintf('Approved backbone and Heading Prediction figures exported; all frozen means checked against current native FIGs.\n');
end
function f=newfig(cm)
f=figure('Visible','off','Color','w','Units','centimeters','Position',[2 2 cm], ...
 'DefaultAxesFontName','Cambria Math','DefaultTextFontName','Cambria Math','DefaultTextInterpreter','none');
end
function styleaxis(a)
set(a,'FontName','Cambria Math','FontSize',7.3,'TickDir','out','TickLength',[.012 .012], ...
 'LineWidth',.6,'Box','off','XColor',[.40 .43 .46],'YColor',[.40 .43 .46], ...
 'GridColor',[.84 .87 .89],'GridAlpha',.65,'Layer','bottom');
end
function ann(f,pos,str,sz,bold,color)
if nargin<6,color=[.12 .16 .20];end
weight='normal';if bold,weight='bold';end
annotation(f,'textbox',pos,'String',str,'LineStyle','none','FontName','Cambria Math', ...
 'FontSize',sz,'FontWeight',weight,'Color',color,'Margin',0,'Interpreter','none');
end
function export_all(f,out,name)
drawnow;savefig(f,fullfile(out,[name '.fig']));
exportgraphics(f,fullfile(out,[name '.pdf']),'ContentType','vector','BackgroundColor','white','Padding',8);
exportgraphics(f,fullfile(out,[name '.png']),'Resolution',300,'BackgroundColor','white','Padding',8);
print(f,fullfile(out,[name '.svg']),'-dsvg','-painters');
end
function a=axis_for(f,label)
a=[];axesList=findall(f,'Type','axes');
for candidate=axesList',if strcmp(candidate.YLabel.String,label),a=candidate;break;end,end
assert(~isempty(a));
end
