% GSR preprocessing and feature extraction
% This code is used for GSR preprocessing and set concatenating.
% Author: ltq, zwq
% Toolbox: EEGLAB
% Date: 2023.2.8
%% data preprocessing & enframing
addpath D:\\eegprocess\\readmultibdfdata
label = {'brake','turn','change','throttle','stable'};
for dataset_num = 1:30
    for event_num = 1:4
        path = strcat('D:\\datasets\\set',num2str(dataset_num),'\\',char(label(event_num)),'\\GSR\\');
        if exist(strcat(path,'merged.set'),'file')
            EEG = pop_loadset(strcat(path,'merged.set'));
        else
            EEG = readbdfdata({'data.bdf' 'evt.bdf'},path);
        end
        EEG.etc.eeglabvers = '2021.1'; % this tracks which version of EEGLAB is being used, you may ignore it
        EEG.setname = 'GSR';
        EEG = eeg_checkset( EEG );
        EEG = pop_eegfiltnew(EEG, 'locutoff',0.5,'hicutoff',100,'plotfreqz',0);
        EEG = eeg_checkset( EEG );
        EEG = pop_resample( EEG, 200);
        EEG = eeg_checkset( EEG );
        pop_saveset( EEG, 'filename',strcat(char(label(event_num)),'_postconpro'),'filepath',path);
    end
end
%% Extracting specific markers from the preprocessed data
addpath D:\\
label = {'brake','turn','change','throttle','stable'};
for dataset_num = 1:30
    for event_num = 1:5
        path = strcat('D:\\datasets\\set',num2str(dataset_num),'\\',char(label(event_num)),'\\GSR\\');
        if event_num == 5
            path = strcat('D:\\datasets\\set',num2str(dataset_num),'\\turn\\GSR\\');
            EEG = pop_loadset(strcat(path,'\\turn_postconpro.set'));
        else
            EEG = pop_loadset(strcat(path,'\\',char(label(event_num)),'_postconpro.set'));
        end
        EEG = eeg_checkset( EEG );
            % 前车急刹、侧面cut和行人横穿：{139 141 145}
            % 左转弯、右转弯：{125 127}
            % 左变道、右变道：{129 131}
            % 超车、前方拥堵：{137 143}
            % 稳定行驶：{133}
        switch event_num
            case 1
                mark = [139 141 145];
                markset = {'139' '141' '145'};

            case 2
                mark = [125 127];
                markset = {'125' '127'};

            case 3
                mark = [129 131];
                markset = {'129' '131'};

            case 4
                mark = [137 143];
                markset = {'137' '143'};

            case 5
                mark = [133];
                markset = {'133'};
        end
        EEG.event = select_mark(EEG.event,mark);
        EEG = eeg_checkset( EEG );
        EEG = pop_epoch( EEG, markset, [-1 3], 'newname', 'epochs', 'epochinfo', 'yes');
        EEG = eeg_checkset( EEG );
        EEG = pop_rmbase( EEG, [-1000 0] ,[]);
        EEG = eeg_checkset( EEG );
        if event_num == 5
            path = strcat('D:\\datasets\\set',num2str(dataset_num),'\\stable\\GSR\\');
        end
        pop_saveset( EEG, 'filename',strcat('preICA_',char(label(event_num)),'_event.set'),'filepath',path);
    end
end
%% 提取和拼接stable
addpath D:\\
for dataset_num = 1:9
    mkdir(strcat('D:\\set',num2str(dataset_num),'\\stable\\GSR\\'));
    for event_num = 2:2
            path = strcat('D:\\set',num2str(dataset_num));
            EEG = pop_loadset(strcat(path,'\\turn\\GSR\\turn_postconpro.set'));
            EEG.etc.eeglabvers = '2021.1'; % this tracks which version of EEGLAB is being used, you may ignore it
            EEG.setname = 'eeg';
            EEG = eeg_checkset( EEG );
            EEG.event = select_mark(EEG.event,[133]);
            EEG = eeg_checkset( EEG );
            EEG = pop_epoch( EEG, {'133'}, [-1 3], 'newname', 'epochs', 'epochinfo', 'yes');
            EEG = eeg_checkset( EEG );
            EEG = pop_rmbase( EEG, [-1000 0] ,[]);
            EEG = eeg_checkset( EEG );
    end
    pop_saveset( EEG, 'filename',strcat('preICA_stable_event.set'),'filepath',strcat('D:\\set',num2str(dataset_num),'\\stable\\GSR\\'));
end
%% 拼接数据集pop_mergeset
label = {'brake','turn','change','throttle','stable'};
for dataset_num = 1:30
    EEG = pop_loadset(strcat('D:\\datasets\\set',num2str(dataset_num),'\\',char(label(1)),'\\GSR\\preICA_',char(label(1)),'_event.set'));
    for event_num = 2:5
        path = strcat('D:\\datasets\\set',num2str(dataset_num),'\\',char(label(event_num)),'\\GSR\\');
        filename = strcat('preICA_',char(label(event_num)),'_event.set');
        EEG = pop_mergeset(EEG, pop_loadset(strcat(path,'\\',filename)));%拼接数据集
        EEG = eeg_checkset( EEG );
    end
    pop_saveset( EEG, 'filename',strcat('preICA_',num2str(dataset_num),'_GSR.set'),'filepath',strcat('D:\\datasets\\set',num2str(dataset_num)));
end
%% 绘图，先按照事件合并数据。
    for event_num = 1:5
        path = strcat('D:\\datasets\\set1\\',char(label(event_num)),'\\GSR\\');
        EEG = pop_loadset(strcat(path,'preICA_',char(label(event_num)),'_event.set'));
        for dataset_num = 2:30
            path = strcat('D:\\datasets\\set',num2str(dataset_num),'\\',char(label(event_num)),'\\GSR\\');
            filename = strcat('preICA_',char(label(event_num)),'_event.set');
            EEG = pop_mergeset(EEG, pop_loadset(strcat(path,'\\',filename)));%拼接数据集
            EEG = eeg_checkset( EEG );
        end
        pop_saveset( EEG, 'filename',strcat('preICA_',char(label(event_num)),'_GSR.set'),'filepath','D:\\datasets\\');
    end
    %% 画图
%     tic;
%     for event_num = 1:5
%         EEG = pop_loadset(strcat('D:\\datasets\\preICA_',char(label(event_num)),'_GSR.set'));
%         for i = 1:size(EEG.data,3)
%             subplot(2,3,event_num);
%             plot(EEG.data(1,:,i));
%             hold on;
%         end
%         xlabel(char(label(event_num)));
%         ylabel('value');
%         axis([0 800 -10 10])
%     end
%     toc;
label = {'brake','turn','change','throttle','stable'};
figure;
channel_num = 1;
for event_num = 1:4
     EEG = pop_loadset(strcat('D:\\datasets\\preICA_',char(label(event_num)),'_GSR.set'));
     subplot(4,1,event_num);
     for i=1:20%前20个样本
         plot(EEG.data(1,:,30*i)+i*0.1);
         title('GSR Waveforms');
         ylabel(char(label(event_num)))
         xlabel('GSR');
         hold on;
     end
end
%% 手动删除坏帧
    eeglab


%% yuanshi feichu 

%% 提取指定的事件Marker
label = {'brake','turn','change'};
for dataset_num = 13:15
    path = strcat('D:\\20220331\\',num2str(dataset_num),'_3\\');
    % readbdfdata
    EEG = readbdfdata({'data.bdf' 'evt.bdf'},path);
    EEG.etc.eeglabvers = '2021.1'; % this tracks which version of EEGLAB is being used, you may ignore it
    EEG.setname = 'GSR';
    EEG = eeg_checkset( EEG );
    EEG = pop_eegfiltnew(EEG, 'locutoff',0.5,'hicutoff',100,'plotfreqz',0);
    EEG = eeg_checkset( EEG );
    EEG = pop_saveset( EEG, 'filename','filter','filepath',path);
    EEG = eeg_checkset( EEG );
    EEG = pop_resample( EEG, 200);
    EEG = eeg_checkset( EEG );
    pop_saveset(EEG,'filename','postconpro.set','filepath',path);
     for event_num = 1:3

            % 前车急刹、侧面cut和行人横穿：{139 141 145}
            % 左转弯、右转弯：{125 127}
            % 左变道、右变道：{129 131}
        switch event_num
            case 1
                mark = [139 141 145];
                markset = {'139' '141' '145'};
                
            case 2
                mark = [125 127];
                markset = {'125' '127'};
                
            case 3
                mark = [129 131];
                markset = {'129' '131'};
                
        end
        EEG.event = select_mark(EEG.event,mark);
        EEG = eeg_checkset( EEG );
        EEG = pop_epoch( EEG, markset, [-1 1.5], 'newname', 'epochs', 'epochinfo', 'yes');
        EEG = eeg_checkset( EEG );
        % EEG = pop_rmbase( EEG, [-500 0] ,[]);
        EEG = eeg_checkset( EEG );
        pop_saveset( EEG, 'filename',strcat(char(label(event_num)),'_event.set'),'filepath',path);
     end
end