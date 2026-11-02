function report=layout_result_labels(f)
% Synchronize labels with current bars and place them above their complete
% local bar/curve/CI envelope. No data layers or uncertainty limits change.
labels={'SR (%)','NE (m)','SPL (%)','OR (%)','TL (m)'};
aa=findall(f,'Type','axes');report=struct;
for k=1:5
    a=[];for candidate=aa',if strcmp(candidate.YLabel.String,labels{k}),a=candidate;break;end,end
    assert(~isempty(a));
    b=findall(a,'Type','bar');assert(numel(b)==1);
    x=b.XData(:)';y=b.YData(:)';
    numeric=gobjects(0);tx=findall(a,'Type','text');
    for t=tx'
        value=string(t.String);
        if isscalar(value) && isfinite(str2double(value))
            t.Units='data';numeric(end+1)=t;
        elseif startsWith(value,'(')
            t.Units='normalized';t.Position=[.015 .985 0];
            t.VerticalAlignment='top';t.HorizontalAlignment='left';
        end
    end
    assert(numel(numeric)==numel(x),'Expected one numeric label per bar.');
    [~,order]=sort(arrayfun(@(t)t.Position(1),numeric));numeric=numeric(order);
    [x,order]=sort(x);y=y(order);
    for j=1:numel(x)
        numeric(j).String=sprintf('%.1f',y(j));
    end
    lines=findall(a,'Type','line');errors=findall(a,'Type','errorbar');
    dataTop=max(y);
    for e=errors',dataTop=max([dataTop,e.YData(:)'+e.YPositiveDelta(:)']);end
    for line=lines'
        if numel(unique(line.XData(isfinite(line.XData))))>1
            yy=line.YData(isfinite(line.YData));if ~isempty(yy),dataTop=max([dataTop,yy(:)']);end
        end
    end
    a.YLim=[0,max(a.YLim(2),1.45*dataTop)];
    for t=numeric
        t.Units='normalized';t.HorizontalAlignment='center';t.VerticalAlignment='bottom';
        t.FontSize=22;t.FontWeight='normal';t.BackgroundColor='none';t.Tag='ValueLabel';
        t.Clipping='off';
    end
    for attempt=1:8
        drawnow;pixels=getpixelposition(a,true);xr=diff(a.XLim);yr=diff(a.YLim);
        guardX=4/pixels(3);neighbourGap=2/pixels(3);gap=10/pixels(4);stroke=6/pixels(4);
        boxes=zeros(numel(x),4);records=struct([]);
        for j=1:numel(x)
            t=numeric(j);xn=(x(j)-a.XLim(1))/xr;
            t.Position=[xn .4 0];drawnow;extent=t.Extent;
            lo=a.XLim(1)+(extent(1)-guardX)*xr;
            hi=a.XLim(1)+(extent(1)+extent(3)+guardX)*xr;
            envelope=local_envelope(a,b,lines,errors,lo,hi);
            bottom=(envelope-a.YLim(1))/yr+stroke+gap;
            % Neighbouring long labels are separated vertically if needed.
            for previous=1:j-1
                if extent(1)<boxes(previous,1)+boxes(previous,3)+neighbourGap && ...
                        extent(1)+extent(3)>boxes(previous,1)-neighbourGap && ...
                        bottom<boxes(previous,2)+boxes(previous,4)+gap && ...
                        bottom+extent(4)>boxes(previous,2)-gap
                    bottom=boxes(previous,2)+boxes(previous,4)+gap;
                end
            end
            t.Position=[xn,bottom-(extent(2)-t.Position(2)),0];drawnow;
            boxes(j,:)=t.Extent;
            records(j).text=char(string(t.String));
            records(j).bar_x=x(j);records(j).bar_y=y(j);
            records(j).bbox_normalized=boxes(j,:);
            records(j).envelope_y=envelope;
            records(j).clearance_pixels=(boxes(j,2)-(envelope-a.YLim(1))/yr)*pixels(4)-6;
            assert(records(j).clearance_pixels>=9.99);
        end
        if max(boxes(:,2)+boxes(:,4))<.86,break;end
        a.YLim(2)=a.YLim(2)*1.12;
    end
    assert(max(boxes(:,2)+boxes(:,4))<.86,'Insufficient room for value labels.');
    for j=1:size(boxes,1)
        for q=j+1:size(boxes,1)
            assert(~(boxes(j,1)<boxes(q,1)+boxes(q,3) && boxes(q,1)<boxes(j,1)+boxes(j,3) && ...
                boxes(j,2)<boxes(q,2)+boxes(q,4) && boxes(q,2)<boxes(j,2)+boxes(j,4)));
        end
    end
    report.(strtok(labels{k},' '))=struct('labels',records,'ylim',a.YLim, ...
        'minimum_clearance_pixels',min([records.clearance_pixels]));
end
end

function top=local_envelope(a,b,lines,errors,lo,hi)
x=b.XData(:)';y=b.YData(:)';step=min(diff(sort(x)));if isempty(step),step=1;end
halfWidth=step*b.BarWidth/2;
hit=(x+halfWidth>=lo & x-halfWidth<=hi);top=max([0,y(hit)]);
for line=lines'
    xx=line.XData(:)';yy=line.YData(:)';
    if numel(xx)==1 && xx>=lo && xx<=hi,top=max(top,yy);end
    for s=1:numel(xx)-1
        if any(~isfinite([xx(s:s+1),yy(s:s+1)])),continue;end
        left=max(lo,min(xx(s:s+1)));right=min(hi,max(xx(s:s+1)));
        if right<left,continue;end
        if xx(s)==xx(s+1)
            values=yy(s:s+1);
        else
            values=yy(s)+(yy(s+1)-yy(s))*([left right]-xx(s))/(xx(s+1)-xx(s));
        end
        top=max([top,values]);
    end
end
pixels=getpixelposition(a,true);capRadius=8/pixels(3)*diff(a.XLim);
for e=errors'
    xx=e.XData(:)';upper=e.YData(:)'+e.YPositiveDelta(:)';
    hit=(xx+capRadius>=lo & xx-capRadius<=hi);top=max([top,upper(hit)]);
end
end
