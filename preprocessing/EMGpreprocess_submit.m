% EMG  preprocessing
% This code is used for EMG preprocessing and set concatenating.
% Author: ltq, zwq
% Toolbox: EEGLAB
% Date: 2023.2.8
%% data preprocessing & enframing
addpath D:\\eegprocess\\readmultibdfdata
label = {'brake','turn','change','throttle','stable'};
for dataset_num = 1:30
    for event_num = 1:4
        path = strcat('D:\\datasets\\set',num2str(dataset_num),'\\',char(label(event_num)),'\\EMG\\');
        if exist(strcat(path,'merged.set'),'file')
            EEG = pop_loadset(strcat(path,'merged.set'));
        else
            EEG = readbdfdata({'data.bdf' 'evt.bdf'},path);
        end
        EEG.etc.eeglabvers = '2021.1'; % this tracks which version of EEGLAB is being used, you may ignore it
        EEG.setname='EMG';
        EEG = eeg_checkset( EEG );
        EEG = pop_eegfiltnew(EEG, 'locutoff',15,'hicutoff',90,'plotfreqz',0);
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
    for event_num = 1:4
        path = strcat('D:\\datasets\\set',num2str(dataset_num),'\\',char(label(event_num)),'\\EMG\\');
        if event_num == 5
            path = strcat('D:\\datasets\\set',num2str(dataset_num),'\\turn\\EMG\\');
            EEG = pop_loadset(strcat(path,'\\turn_postconpro.set'));
        else
            EEG = pop_loadset(strcat(path,'\\',char(label(event_num)),'_postconpro.set'));
        end

        EEG.etc.eeglabvers = '2021.1'; % this tracks which version of EEGLAB is being used, you may ignore it
        EEG.setname = 'eeg';
        EEG = eeg_checkset( EEG );
            % brake: front emergency brake, side front cut-in, pedestrian crossing：{139 141 145}
            % turning:  Left-turn sign, Right-turn sign：{125 127}
            % lane changing: Static obstacle on the right/left：{129 131}
            % throttle: overtaking, Congestion relief：{137 143}
            % stable：{133}
        switch event_num %select the mark set of the event
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
            path = strcat('D:\\datasets\\set',num2str(dataset_num),'\\stable\\EMG\\');
        end
        pop_saveset( EEG, 'filename',strcat('preICA_',char(label(event_num)),'_event.set'),'filepath',path);
    end
end
%% use the function pop_mergeset to merge datasets(according to events)
label = {'brake','turn','change','throttle','stable'};
for event_num = 1:5
    EEG = pop_loadset(strcat('D:\\datasets\\set1\\',char(label(event_num)),'\\EMG\\preICA_',char(label(event_num)),'_event.set'));
    for dataset_num = 2:30
            path = strcat('D:\\datasets\\set',num2str(dataset_num),'\\',char(label(event_num)),'\\EMG');
            filename = strcat('preICA_',char(label(event_num)),'_event.set');
            EEG = pop_mergeset(EEG, pop_loadset(strcat(path,'\\',filename)));
            EEG = eeg_checkset( EEG );
    end
    filename = strcat('EMG_',char(label(event_num)),'_event_merged_new.set');
    mkdir('D:\\data\\EMG\\')
    EEG = pop_saveset( EEG, 'filename',filename,'filepath','D:\\data\\EMG\\');
    EEG = eeg_checkset( EEG );
end
%% use EEGLAB to remove bad data manually
eeglab