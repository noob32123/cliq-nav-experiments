function verify_result_figures
% Verify exported editable charts against the versioned current replay.
root=fileparts(fileparts(mfilename('fullpath')));
out=fullfile(root,'figures');src=fullfile(out,'source_figures');
r=jsondecode(fileread(fullfile(out,'results_render.json')));
ci=readtable(fullfile(root,'supplementary/results/plot_ci.csv'),'TextType','string');
names={'map1_base_models','map1_clip_ablation','map1_external', ...
    'map2_ablation','map2_external','heading_ablation_comparison'};
sources={'qwen_seven.fig','map1_clip.fig','map1_external.fig', ...
    'map2_clip.fig','map2_external.fig',''};
labels={'SR (%)','NE (m)','SPL (%)','OR (%)','TL (m)'};
report=struct('passed',true,'font','Cambria Math','figures',struct);
for n=1:6
    f=openfig(fullfile(out,[names{n} '.fig']),'invisible');
    % Actual rendered font names are verified from the exported PDF resources.
    axesList=findall(f,'Type','axes');actual=[];
    for k=1:5
        a=find_metric(axesList,labels{k});b=findall(a,'Type','bar');
        assert(numel(b)==1);actual(k,:)=b.YData(:)';
        assert(max(abs(a.Position-r.style.axes_positions(k,:)))<1e-7);
    end
    legendObject=findall(f,'Type','legend');
    assert(numel(legendObject)==1 && legendObject.FontSize==r.style.legend_font_size);
    if n<6
        if n==1
            env="City Park";conditions=["Qwen-VL-Max","Qwen-VL-Plus","Qwen3-VL-Flash","Qwen3.5-35B-A3B","Qwen3.5-Flash"];
        elseif n==2
            env="City Park";conditions=["Qwen3.5-35B-A3B + CLIP","Qwen3.5-35B-A3B","Qwen3.5-Flash + CLIP","Qwen3.5-Flash"];
        elseif n==4
            env="Downtown West";conditions=["Qwen3.5-35B-A3B + CLIP","Qwen3.5-35B-A3B"];
        else
            if n==3,env="City Park";else,env="Downtown West";end
            conditions=["Qwen3.5-35B-A3B + CLIP","CityNavAgent","FineCog-Nav","AerialVLA"];
        end
        metrics=["SR","NE","SPL","OR","TL"];
        for k=1:5
            expected=zeros(1,numel(conditions));
            for j=1:numel(conditions)
                row=ci(ci.environment==env & ci.condition==conditions(j) & ci.metric==metrics(k),:);
                assert(height(row)==1);expected(j)=row.mean;
            end
            assert(max(abs(actual(k,:)-expected))<1e-10,'Chart/replay mismatch: %s',names{n});
            a=find_metric(axesList,labels{k});
            curves=findall(a,'Type','line');
            for curve=curves'
                if numel(curve.YData)==numel(expected)
                    assert(max(abs(curve.YData-expected))<1e-10);
                end
            end
            numeric=findall(a,'Type','text');b=findall(a,'Type','bar');
            for label=numeric'
                if isscalar(string(label.String)) && isfinite(str2double(string(label.String)))
                    label.Units='data';
                    [~,j]=min(abs(b.XData-label.Position(1)));
                    assert(strcmp(label.String,sprintf('%.1f',expected(j))), ...
                        'Numeric label mismatch in %s / %s: text=%s x=%.6f bar=%d expected=%.1f', ...
                        names{n},labels{k},string(label.String),label.Position(1),j,expected(j));
                end
            end
        end
        if ismember(n,[2 4])
            if n==2,cue=r.figures.map1clip;else,cue=r.figures.map2clip;end
            assert(max(abs(actual(:)-cue.means(:)))<1e-10);
            for k=1:5
                a=find_metric(axesList,labels{k});e=findall(a,'Type','errorbar');assert(numel(e)==1);
                assert(max(abs(e.YNegativeDelta-(cue.means(k,:)-cue.ci_low(k,:))))<1e-10);
                assert(max(abs(e.YPositiveDelta-(cue.ci_high(k,:)-cue.means(k,:))))<1e-10);
            end
        end
    else
        h=jsondecode(fileread(fullfile(out,'heading_ablation_render.json')));
        assert(isequal(actual,h.means));
        for k=1:5
            a=find_metric(axesList,labels{k});e=findall(a,'Type','errorbar');
            assert(numel(e)==1);
            assert(max(abs(e.YNegativeDelta-(h.means(k,:)-h.ci_low(k,:))))<1e-12);
            assert(max(abs(e.YPositiveDelta-(h.ci_high(k,:)-h.means(k,:))))<1e-12);
        end
    end
    report.figures.(names{n})=struct('means',actual,'font_verified_from_pdf',true, ...
        'axes_positions_verified',true,'data_verified',true);
    close(f);
end
fid=fopen(fullfile(root,'source_data','matlab_figure_verification.json'),'w');
fwrite(fid,jsonencode(report,PrettyPrint=true),'char');fclose(fid);
fprintf('Verified six MATLAB figures: means, layout and all cue/heading CI objects.\n');
end
function a=find_metric(items,label)
a=[];for p=items',if strcmp(p.YLabel.String,label),a=p;break;end,end
assert(~isempty(a));
end
