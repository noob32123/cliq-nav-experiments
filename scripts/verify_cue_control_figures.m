function verify_cue_control_figures
% Verify three-condition charts against their complete route-bootstrap input.
root=fileparts(fileparts(mfilename('fullpath')));
ci=readtable(fullfile(root,'supplementary/cue_prior_control/plot_ci.csv'),'TextType','string');
report=jsondecode(fileread(fullfile(root,'figures/results_render.json')));
conditions=["Qwen3.5-35B-A3B + CLIP", ...
    "Qwen3.5-35B-A3B + uniform direction prior","Qwen3.5-35B-A3B"];
environments=["City Park","Downtown West"];
metrics=["SR","NE","SPL","OR","TL"];
labels={'SR (%)','NE (m)','SPL (%)','OR (%)','TL (m)'};
result=struct('passed',true,'backend','MATLAB','version',version,'figures',struct);
for p=1:2
    name=sprintf('map%d_cue_prior_control',p);
    f=openfig(fullfile(root,'figures',[name '.fig']),'invisible');
    axesList=findall(f,'Type','axes');
    for k=1:5
        a=[];
        for candidate=axesList'
            if strcmp(candidate.YLabel.String,labels{k}),a=candidate;break;end
        end
        assert(~isempty(a));
        assert(strcmp(a.FontName,'Cambria Math'));
        assert(max(abs(a.Position-report.style.axes_positions(k,:)))<1e-7);
        bars=findall(a,'Type','bar');errors=findall(a,'Type','errorbar');
        assert(numel(bars)==1 && numel(errors)==1);
        means=zeros(1,3);lows=means;highs=means;
        for j=1:3
            row=ci(ci.environment==environments(p) & ci.condition==conditions(j) & ci.metric==metrics(k),:);
            assert(height(row)==1);
            means(j)=row.mean;lows(j)=row.low;highs(j)=row.high;
        end
        assert(max(abs(bars.YData-means))<1e-10);
        assert(max(abs(errors.YNegativeDelta-(means-lows)))<1e-10);
        assert(max(abs(errors.YPositiveDelta-(highs-means)))<1e-10);
        numeric=findall(a,'Tag','ValueLabel');assert(numel(numeric)==3);
        for t=numeric'
            t.Units='data';[~,j]=min(abs(t.Position(1)-bars.XData));
            assert(strcmp(t.String,sprintf('%.1f',means(j))));
        end
    end
    legendObject=findall(f,'Type','legend');
    assert(numel(legendObject)==1 && strcmp(legendObject.FontName,'Cambria Math'));
    assert(legendObject.FontSize==report.style.legend_font_size);
    result.figures.(name)=struct('all_five_means_verified',true, ...
        'all_fifteen_intervals_verified',true,'original_layout_and_font_verified',true);
    close(f);
end
fid=fopen(fullfile(root,'source_data/cue_control_figure_verification.json'),'w');
fwrite(fid,jsonencode(result,PrettyPrint=true),'char');fclose(fid);
fprintf('Verified two three-condition MATLAB figures: all means, intervals, fonts and axes.\n');
end
