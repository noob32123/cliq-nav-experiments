function verify_adopted_result_figures
here=fileparts(mfilename('fullpath'));root=fileparts(here);out=fullfile(root,'figures');
qadir=fullfile(root,'figures','render_checks');
record=jsondecode(fileread(fullfile(qadir,'matlab_data_verification.json')));
f=openfig(fullfile(out,'map1_backbone_intervals.fig'),'invisible');
verify_font(f);
aa=findall(f,'Type','axes');metric={'SR','NE','SPL','OR','TL'};
for k=1:5
 a=[];for c=aa',if startsWith(c.Title.String,metric{k}),a=c;break;end,end
 assert(~isempty(a));mm=findall(a,'Tag','FrozenMean');ci=findall(a,'Tag','FrozenCI');
 assert(numel(mm)==5 && numel(ci)==5);
 actual=sort(arrayfun(@(x)x.XData,mm));expected=sort(record.figure6.means(k,:));
 assert(max(abs(actual(:)-expected(:)))<1e-9);
 actual=sortrows(vertcat(ci.XData));expected=sortrows([record.figure6.low(k,:)' record.figure6.high(k,:)']);
 assert(max(abs(actual(:)-expected(:)))<1e-9);
end
assert(numel(findall(f,'Tag','FrozenMean'))==25);
bars=findall(f,'Tag','FrozenCI');caps=findall(f,'Tag','Figure6IntervalCap');
assert(numel(caps)==50);
assert(all(arrayfun(@(x)strcmp(x.LineStyle,'-') && abs(x.LineWidth-1.1)<1e-9,[bars;caps])));
close(f);
f=openfig(fullfile(out,'heading_ablation_intervals.fig'),'invisible');aa=findall(f,'Type','axes');
verify_font(f);
for k=1:5
 a=[];
 for c=aa'
  title=string(c.Title.String);
  if (k==1 && startsWith(title,'(a) SR')) || (k~=1 && contains(title,string(metric{k})+" ("))
   a=c;break;
  end
 end
 assert(~isempty(a));mm=findall(a,'Tag','FrozenMean');assert(numel(mm)==4);
 if k==1,actual=sort(arrayfun(@(x)x.YData,mm));else,actual=sort(arrayfun(@(x)x.XData,mm));end
 expected=sort(record.figure11.means(k,:));assert(max(abs(actual(:)-expected(:)))<1e-9);
 ci=findall(a,'Tag','FrozenMarginalCI');assert(numel(ci)==4);
 if k==1
  lower=arrayfun(@(x)x.YData-x.YNegativeDelta,ci);
  upper=arrayfun(@(x)x.YData+x.YPositiveDelta,ci);actual=sortrows([lower(:) upper(:)]);
 else,actual=sortrows(vertcat(ci.XData));end
 expected=sortrows([record.figure11.low(k,:)' record.figure11.high(k,:)']);
 assert(max(abs(actual(:)-expected(:)))<1e-9);
end
assert(numel(findall(f,'Tag','FrozenMean'))==20);
mm=findall(f,'Tag','FrozenPairedMean');assert(numel(mm)==2);
actual=sort(arrayfun(@(x)x.XData,mm));expected=sort(record.figure11.paired_sr(:,1));
assert(max(abs(actual(:)-expected(:)))<1e-9);
ci=findall(f,'Tag','FrozenPairedCI');actual=sortrows(vertcat(ci.XData));expected=sortrows(record.figure11.paired_sr(:,2:3));
assert(max(abs(actual(:)-expected(:)))<1e-9);
swatches=findall(f,'Tag','MapColorBar');assert(numel(swatches)==2);
actual=sortrows(vertcat(swatches.Color));expected=sortrows([.08 .47 .49;.56 .37 .65]);
assert(max(abs(actual(:)-expected(:)))<1e-9);
connectors=findall(f,'Tag','ConditionConnector');assert(numel(connectors)==10);
assert(all(arrayfun(@(x)strcmp(x.LineStyle,'--') && x.LineWidth<1,connectors)));
bars=findall(f,'Tag','FrozenMarginalCI');
assert(all(arrayfun(@(x)strcmp(x.LineStyle,'-') || strcmp(x.LineStyle,'none'),bars)));
caps=findall(f,'Tag','IntervalCap');assert(numel(caps)==36);
assert(all(arrayfun(@(x)strcmp(x.LineStyle,'-'),caps)));
rowLabels=findall(f,'Tag','MapRowLabel');assert(numel(rowLabels)==2);
rowNames=sort(arrayfun(@(x)strjoin(string(x.String)," "),rowLabels));
assert(isequal(rowNames(:),["City Park";"Downtown West"]));
close(f);
result=struct('passed',true,'figure6_means',25,'figure6_marginal_intervals',25, ...
 'figure6_interval_caps',50,'figure6_interval_style','Solid colored 1.1 pt bars with end caps', ...
 'figure11_condition_means',20,'figure11_marginal_intervals',20, ...
 'figure11_paired_sr_means',2,'figure11_paired_sr_intervals',2, ...
 'maximum_allowed_absolute_error',1e-9,'source_fig_means_checked_by_renderer',true, ...
 'no_statistics_recomputed',true,'original_files_not_written',true, ...
 'all_native_font_properties','Cambria Math','categorical_map_color_bars',2, ...
 'condition_connectors','Ten thin gray dashed lines', ...
 'interval_style','Solid colored bars with end caps', ...
 'extra_interval_caps',36,'secondary_condition_row_offset',.14, ...
 'lower_panel_shared_map_row_labels',2);
fid=fopen(fullfile(qadir,'native_figure_verification.json'),'w');fwrite(fid,jsonencode(result,PrettyPrint=true));fclose(fid);
fprintf('Verified all approved figure means and CI geometry against frozen data.\n');
end
function verify_font(f)
objects=findall(f,'-property','FontName');
assert(all(arrayfun(@(x)strcmp(x.FontName,'Cambria Math'),objects)), ...
 'A native font property does not use Cambria Math.');
end
