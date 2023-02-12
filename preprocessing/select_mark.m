function list = select_mark(list,event)
    i=1;
    while i~=length(list)+1
        if ismember(str2double(list(i).type),event)==0
            list(i)=[];
        else
            i=i+1;
        end
    end