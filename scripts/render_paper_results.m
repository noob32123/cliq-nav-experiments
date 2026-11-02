function render_paper_results(selection)
% Paper results in Cambria Math using the existing external-comparison layout.
% Read frozen MATLAB data objects; the only exclusion is the requested removal
% of the two CLIP-assisted configurations from the base-model comparison.
if nargin<1, selection='all'; end
here=fileparts(mfilename('fullpath')); root=fileparts(here);
out=fullfile(root,'figures'); inputs=fullfile(out,'source_figures');
assert(any(strcmp(listfonts,'Cambria Math')),'Cambria Math is not installed.');
style=read_style(fullfile(inputs,'map1_external.fig'));
[allModels,allColors]=read_data(fullfile(inputs,'qwen_seven.fig'));
keep=[1 2 3 5 7];
baseNames={'Qwen-VL-Max','Qwen-VL-Plus','Qwen3-VL-Flash', ...
    'Qwen3.5-35B-A3B','Qwen3.5-Flash'};
jobs={};
jobs{end+1}=job('base','map1_base_models',allModels(:,keep),allColors(keep,:),baseNames,true,false);
[v,c]=read_data(fullfile(inputs,'map1_clip.fig'));
jobs{end+1}=job('map1clip','map1_clip_ablation',v,c, ...
    {'35B-A3B + CLIP','35B-A3B (No cue)','Flash + CLIP','Flash (No cue)'},false,true);
[v,c]=read_data(fullfile(inputs,'map1_external.fig'));
jobs{end+1}=job('map1external','map1_external',v,c, ...
    {'CLiQ-Nav (Ours)','CityNavAgent','FineCog-Nav','AerialVLA'},true,false);
[v,c]=read_data(fullfile(inputs,'map2_clip.fig'));
jobs{end+1}=job('map2clip','map2_ablation',v,c, ...
    {'35B-A3B + CLIP','35B-A3B (No cue)'},false,false);
[v,c]=read_data(fullfile(inputs,'map2_external.fig'));
jobs{end+1}=job('map2external','map2_external',v,c, ...
    {'CLiQ-Nav (Ours)','CityNavAgent','FineCog-Nav','AerialVLA'},true,false);
% Keep the original FIGs as style/data-provenance inputs. Current scores come
% from the versioned replay, including the revised seven Downtown radii.
ci=readtable(fullfile(root,'supplementary','results','plot_ci.csv'),'TextType','string');
original=readtable(fullfile(root,'supplementary','results_original_6m','plot_ci.csv'),'TextType','string');
for n=1:numel(jobs)
    item=jobs{n};
    switch item.key
        case 'base'
            environment="City Park";conditions=string(baseNames);
        case 'map1clip'
            environment="City Park";
            conditions=["Qwen3.5-35B-A3B + CLIP","Qwen3.5-35B-A3B", ...
                        "Qwen3.5-Flash + CLIP","Qwen3.5-Flash"];
        case 'map2clip'
            environment="Downtown West";
            conditions=["Qwen3.5-35B-A3B + CLIP","Qwen3.5-35B-A3B"];
        otherwise
            if strcmp(item.key,'map1external'),environment="City Park";
            else,environment="Downtown West";end
            conditions=["Qwen3.5-35B-A3B + CLIP","CityNavAgent","FineCog-Nav","AerialVLA"];
    end
    metricNames=["SR","NE","SPL","OR","TL"];
    for k=1:5
        for j=1:numel(conditions)
            row=ci(ci.environment==environment & ci.condition==conditions(j) & ci.metric==metricNames(k),:);
            assert(height(row)==1);
            old=original(original.environment==environment & original.condition==conditions(j) & original.metric==metricNames(k),:);
            assert(height(old)==1 && abs(item.values(k,j)-old.mean)<.051,'Original FIG and original-radius replay differ.');
            item.values(k,j)=row.mean;
            if ismember(item.key,{'map1clip','map2clip'})
                item.low(k,j)=row.low;item.high(k,j)=row.high;
            end
        end
    end
    jobs{n}=item;
end
T=readtable(fullfile(root,'source_data','heading_ablation_summary.csv'),'TextType','string');
groups=["map1_ablated","map1_full","map2_ablated","map2_full"];
metrics=["SR","NE","SPL","OR","TL"];
means=zeros(5,4); lows=means; highs=means;
for k=1:5
    for j=1:4
        row=T(T.group==groups(j) & T.metric==metrics(k),:);
        assert(height(row)==1 && row.episodes==10*row.routes);
        means(k,j)=row.mean; lows(k,j)=row.ci_low; highs(k,j)=row.ci_high;
    end
end
assert(all(lows(:)<=means(:) & highs(:)>=means(:)));
h=job('heading','heading_ablation_comparison',means, ...
    [170 190 210;55 85 130;240 210 170;178 120 45]/255, ...
    {'Map1: Ablated','Map1: Unablated', ...
     'Map2: Ablated','Map2: Unablated'},false,true);
h.low=lows; h.high=highs; jobs{end+1}=h;
% Optional completed uniform-prior control: inherit the same MATLAB source
% layout, fonts and palette. Never substitute illustrative or partial values.
priorPath=fullfile(root,'supplementary','cue_prior_control','plot_ci.csv');
if isfile(priorPath)
    prior=readtable(priorPath,'TextType','string');
    priorConditions=["Qwen3.5-35B-A3B + CLIP","Qwen3.5-35B-A3B + uniform direction prior","Qwen3.5-35B-A3B"];
    priorEnvironments=["City Park","Downtown West"];
    priorNames={'Actual CLIP cue','Random cue','No cue'};
    for p=1:2
        v=zeros(5,3);lo=v;hi=v;
        for k=1:5
            for j=1:3
                row=prior(prior.environment==priorEnvironments(p) & prior.condition==priorConditions(j) & prior.metric==metrics(k),:);
                assert(height(row)==1,'Incomplete uniform-prior result grid.');
                v(k,j)=row.mean;lo(k,j)=row.low;hi(k,j)=row.high;
            end
        end
        item=job('cueprior',sprintf('map%d_cue_prior_control',p),v, ...
            [55 85 130;170 180 190;210 180 150]/255,priorNames,false,false);
        item.low=lo;item.high=hi;jobs{end+1}=item;
    end
end
report=struct('font','Cambria Math','style',style, ...
    'base_model_source_indices',keep,'excluded_source_indices',[4 6], ...
    'exclusion_reason','CLIP-assisted groups moved to the separate Map1 CLIP ablation', ...
    'figures',struct);
if ~strcmp(selection,'all') && isfile(fullfile(out,'results_render.json'))
    old=jsondecode(fileread(fullfile(out,'results_render.json')));report.figures=old.figures;
end
for n=1:numel(jobs)
    item=jobs{n};
    if ~strcmp(selection,'all') && ~strcmp(selection,item.key), continue; end
    if ismember(item.key,{'map1clip','map1external','map2clip','map2external'})
        record=restyle_existing(item,style,out,inputs);
    else
        record=draw_result(item,style,out);
    end
    reportKey=item.key;
    if strcmp(item.key,'cueprior'),reportKey=item.name;end
    report.figures.(reportKey)=record;
    if strcmp(item.key,'heading')
        record.backend='MATLAB'; record.version=version;
        record.group_order=cellstr(groups); record.metric_order=cellstr(metrics);
        record.means=means;record.ci_low=lows;record.ci_high=highs;
        record.seed=20260823;record.interval='95% route-clustered percentile bootstrap, 10000 resamples';
        write_json(fullfile(out,'heading_ablation_render.json'),record);
    end
end
write_json(fullfile(out,'results_render.json'),report);
fprintf('Exported selection %s in Cambria Math.\n',selection);
end
function s=job(key,name,values,colors,names,curve,paired)
s=struct('key',key,'name',name,'values',values,'colors',colors, ...
    'names',{names},'curve',curve,'paired',paired,'low',[],'high',[]);
assert(size(values,1)==5 && size(values,2)==numel(names));
assert(all(isfinite(values(:))));
end
function style=read_style(path)
f=openfig(path,'invisible'); names={'SR (%)','NE (m)','SPL (%)','OR (%)','TL (m)'};
style=struct('canvas_pixels',f.Position(3:4),'axes_positions',zeros(5,4), ...
    'tick_font_size',zeros(1,5),'axis_label_size',zeros(1,5));
axesList=findall(f,'Type','axes');
for k=1:5
    a=metric_axis(axesList,names{k});style.axes_positions(k,:)=a.Position;
    style.tick_font_size(k)=a.FontSize;style.axis_label_size(k)=a.YLabel.FontSize;
end
lgd=findall(f,'Type','legend'); assert(numel(lgd)==1);
style.legend_position=lgd.Position;style.legend_font_size=lgd.FontSize;
style.legend_row_height=lgd.Position(4)/numel(lgd.String);
style.legend_token_size=lgd.ItemTokenSize;
close(f);
end
function [values,colors]=read_data(path)
f=openfig(path,'invisible'); names={'SR (%)','NE (m)','SPL (%)','OR (%)','TL (m)'};
axesList=findall(f,'Type','axes');values=[];colors=[];
for k=1:5
    a=metric_axis(axesList,names{k}); b=findall(a,'Type','bar');
    assert(numel(b)==1);values(k,:)=b.YData(:)';
    if k==1,colors=b.CData;end
end
close(f);
end
function a=metric_axis(items,label)
a=[];
for p=items',if strcmp(p.YLabel.String,label),a=p;break;end,end
assert(~isempty(a),'Missing metric axis %s',label);
end
function record=draw_result(item,style,out)
f=figure('Visible','off','Color','w','Position',[80 60 style.canvas_pixels], ...
    'Renderer','painters','Name',item.name);
labels={'SR (%)','NE (m)','SPL (%)','OR (%)','TL (m)'};
x=1:numel(item.names);if item.paired,x=[1 2 4 5];end
for k=1:5
    a=axes('Parent',f,'Position',style.axes_positions(k,:));
    y=item.values(k,:); b=bar(a,x,y,.55,'FaceColor','flat','EdgeColor','none');
    b.CData=item.colors; b.ShowBaseLine='off';hold(a,'on');
    if item.curve
        plot(a,x,y,'-o','Color',[.1 .1 .1],'LineWidth',1.5, ...
            'MarkerSize',6,'MarkerFaceColor','w');
    end
    spread=max(y)-min(y); offset=max(spread*.10,.045*max(y));
    labelY=y+offset;
    if ~isempty(item.low)
        errorbar(a,x,y,y-item.low(k,:),item.high(k,:)-y,'k', ...
            'LineStyle','none','LineWidth',1.3,'CapSize',7,'Marker','none');
        offset=max(.04*max(item.high(k,:)),.5);
        labelY=item.high(k,:)+offset;
    end
    top=max([max(y)*1.22,max(labelY)/.78,1]);
    if k==4,top=max(100,top);end
    if item.paired
        plot(a,[3 3],[0 top],'--','Color',[.65 .65 .65],'LineWidth',.8);
    end
    for j=1:numel(x)
        text(a,x(j),labelY(j),sprintf('%.1f',y(j)), ...
            'HorizontalAlignment','center','VerticalAlignment','bottom', ...
            'FontName','Cambria Math','FontSize',22,'Interpreter','none');
    end
    xlim(a,[min(x)-.6 max(x)+.6]);ylim(a,[0 top]);
    set(a,'XTick',x,'XTickLabel',{},'FontName','Cambria Math', ...
        'FontSize',style.tick_font_size(k),'LineWidth',.85, ...
        'Box','on','TickDir','out','XColor',[.25 .25 .25],'YColor',[.25 .25 .25]);
    ylabel(a,labels{k},'FontName','Cambria Math','FontSize',style.axis_label_size(k), ...
        'Units','normalized','Position',[-.115 .5 0]);
    grid(a,'on');a.GridAlpha=.30;a.GridLineStyle='-';a.Layer='bottom';
    if strcmp(item.key,'base') && k==5,ylim(a,[0 160]);yticks(a,0:40:160);end
    text(a,.015,.98,sprintf('(%c)',char(96+k)),'Units','normalized', ...
        'FontName','Cambria Math','FontSize',24,'FontWeight','bold', ...
        'VerticalAlignment','top','Interpreter','none');
end
legendAx=axes('Parent',f,'Position',[.6869 .0475 .2597 .4225]);
hold(legendAx,'on'); handles=gobjects(1,numel(item.names));
for j=1:numel(item.names)
    handles(j)=patch(legendAx,NaN,NaN,item.colors(j,:),'EdgeColor','none');
end
lgd=legend(legendAx,handles,item.names,'Location','west','Orientation','vertical', ...
    'Box','off','FontName','Cambria Math','FontSize',style.legend_font_size, ...
    'Interpreter','none');
lgd.AutoUpdate='off';lgd.ItemTokenSize=style.legend_token_size;
axis(legendAx,'off');drawnow;
height=style.legend_row_height*numel(item.names);
if strcmp(item.key,'heading')
    lgd.Title.String='Heading Prediction';
    lgd.Title.FontName='Cambria Math';
    lgd.Title.FontSize=style.legend_font_size;
    lgd.Title.FontWeight='normal';
    height=height+style.legend_row_height;
end
center=style.legend_position(2)+style.legend_position(4)/2;
lgd.Position=[style.legend_position(1),center-height/2,style.legend_position(3),height];
drawnow;
clearance=layout_result_labels(f);
record=struct('label_clearance',clearance,'font','Cambria Math','means',item.values,'ci_low',item.low, ...
    'ci_high',item.high,'group_order',{item.names},'axes_positions',style.axes_positions, ...
    'legend_position',lgd.Position,'legend_row_height',style.legend_row_height, ...
    'legend_title',lgd.Title.String);
savefig(f,fullfile(out,[item.name '.fig']));
export_result_pdf(f,fullfile(out,[item.name '.pdf']));
exportgraphics(f,fullfile(out,[item.name '.png']),'Resolution',600,'BackgroundColor','white','Units','points','Padding',6);
print(f,fullfile(out,[item.name '.svg']),'-dsvg','-painters');
close(f);
end
function record=restyle_existing(item,style,out,inputs)
files=struct('map1clip','map1_clip.fig','map1external','map1_external.fig', ...
    'map2clip','map2_clip.fig','map2external','map2_external.fig');
f=openfig(fullfile(inputs,files.(item.key)),'invisible');
f.Position=[80 60 style.canvas_pixels];f.Color='w';
objects=findall(f,'-property','FontName');set(objects,'FontName','Cambria Math');
objects=findall(f,'-property','FontWeight');set(objects,'FontWeight','normal');
objects=findall(f,'-property','Interpreter');set(objects,'Interpreter','none');
labels={'SR (%)','NE (m)','SPL (%)','OR (%)','TL (m)'};
axesList=findall(f,'Type','axes');
for k=1:5
    a=metric_axis(axesList,labels{k});a.Parent=f;a.Units='normalized';a.Position=style.axes_positions(k,:);
    a.FontSize=style.tick_font_size(k);a.YLabel.FontSize=style.axis_label_size(k);
    a.XTickLabel={};a.XLabel.String='';
    a.YLabel.Units='normalized';a.YLabel.Position=[-.115 .5 0];
    b=findall(a,'Type','bar');assert(numel(b)==1);
    b.YData=item.values(k,:);
    curves=findall(a,'Type','line');
    for curve=curves'
        if numel(curve.YData)==numel(b.YData) && isequal(curve.XData,b.XData)
            curve.YData=b.YData;
        end
    end
    tt=findall(a,'Type','text'); high=a.YLim(2);
    for t=tt'
        if startsWith(string(t.String),'(')
            t.Units='normalized';t.Position=[.015 .95 0];t.FontSize=24;
            t.VerticalAlignment='top';t.FontWeight='bold';
        elseif isscalar(string(t.String)) && isfinite(str2double(string(t.String)))
            t.Units='data';
            [~,index]=min(abs(b.XData-t.Position(1)));
            t.String=sprintf('%.1f',item.values(k,index));
            t.FontSize=22;t.Units='data';high=max(high,t.Position(2)/.78);
        end
    end
    a.YLim=[0 high];
    if ~isempty(item.low)
        hold(a,'on');
        delete(findall(a,'Type','errorbar'));
        errorbar(a,b.XData,b.YData,b.YData-item.low(k,:),item.high(k,:)-b.YData, ...
            'k','LineStyle','none','LineWidth',1.3,'CapSize',7,'Marker','none');
        % Existing label strings keep the source figure's one-decimal display.
        a.YLim=[0,max(high,1.45*max(item.high(k,:)))];
    end
end
for a=axesList',if isempty(a.YLabel.String),axis(a,'off');end,end
lgd=findall(f,'Type','legend');assert(numel(lgd)==1);
lgd.String=item.names;
lgd.FontName='Cambria Math';lgd.FontSize=style.legend_font_size;
lgd.ItemTokenSize=style.legend_token_size;lgd.Box='off';
drawnow;h=style.legend_row_height*numel(lgd.String);
center=style.legend_position(2)+style.legend_position(4)/2;
lgd.Units='normalized';lgd.Position=[style.legend_position(1),center-h/2,style.legend_position(3),h];
drawnow;
clearance=layout_result_labels(f);
record=struct('label_clearance',clearance,'font','Cambria Math','means',item.values,'source_fig',files.(item.key), ...
    'group_order',{item.names}, ...
    'ci_low',item.low,'ci_high',item.high, ...
    'reuse','existing MATLAB style; original FIG data checked against original-radius replay; current bars and labels use revised replay; cue whiskers are marginal 95% CIs', ...
    'axes_positions',style.axes_positions,'legend_position',lgd.Position);
savefig(f,fullfile(out,[item.name '.fig']));
export_result_pdf(f,fullfile(out,[item.name '.pdf']));
exportgraphics(f,fullfile(out,[item.name '.png']),'Resolution',600,'BackgroundColor','white','Units','points','Padding',6);
print(f,fullfile(out,[item.name '.svg']),'-dsvg','-painters');
close(f);
end
function write_json(path,record)
pending=[path '.pending'];
fid=fopen(pending,'w');assert(fid~=-1);fwrite(fid,jsonencode(record,PrettyPrint=true),'char');fclose(fid);
[ok,message]=movefile(pending,path,'f');assert(ok,message);
end

function export_result_pdf(f,path)
% Tight vector export with room for complete tick and label glyphs.
exportgraphics(f,path,'ContentType','vector','BackgroundColor','white', ...
    'Units','points','Padding',6);
end
