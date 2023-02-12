% EEG and ECG preprocessing
% This code is used for EEG and ECG preprocessing and set concatenating.
% Author: ltq, zwq
% Toolbox: EEGLAB
% Date: 2023.2.8
%% data preprocessing & enframing
addpath D:\\eegprocess\\readmultibdfdata
label = {'brake','turn','change','throttle','stable'};
for dataset_num = 1:30  %30 subjects for all
    for event_num = 1:4    %4 kinds of events
        path = strcat('D:\\datasets\\set',num2str(dataset_num),'\\',char(label(event_num)),'\\EEG\\');
        if exist(strcat(path,'merged.set'),'file')
            EEG = pop_loadset(strcat(path,'merged.set'));
        else
            EEG = readbdfdata({'data.bdf' 'evt.bdf'},path);
        end
        EEG.etc.eeglabvers = '2021.1'; % this tracks which version of EEGLAB is being used, you may ignore it
        EEG.setname = 'eeg';
        EEG = eeg_checkset( EEG );
        EEG = pop_chanedit(EEG, 'lookup','D:\\Program Files\\MATLAB\\R2021a\\toolbox\\eeglab2021.1\\plugins\\dipfit\\standard_BEM\\elec\\standard_1005.elc');
        EEG = eeg_checkset( EEG );
        EEG = pop_select( EEG, 'nochannel',{'ECG','HEOR','HEOL','VEOU','VEOL'});
        EEG = eeg_checkset( EEG );
        EEG = pop_reref( EEG, []);
        EEG = eeg_checkset( EEG );
        EEG = pop_eegfiltnew(EEG, 'locutoff',0.5,'hicutoff',40,'plotfreqz',0);
        EEG = eeg_checkset( EEG );
        % EEG = pop_resample( EEG, 200);
        % EEG = eeg_checkset( EEG ); 
        pop_saveset( EEG, 'filename',strcat(char(label(event_num)),'_postconpro'),'filepath',path);
    end
end
%% Extracting specific markers from the preprocessed data
addpath D:\\
label = {'brake','turn','change','throttle','stable'};
for dataset_num = 1:30
    for event_num = 1:5
        path = strcat('D:\\datasets\\set',num2str(dataset_num),'\\',char(label(event_num)),'\\EEG\\');
        if event_num == 5  %extracting stable events
            path = strcat('D:\\datasets\\set',num2str(dataset_num),'\\turn\\EEG\\');
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
        switch event_num %select the mark set of the specific event
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
        EEG = pop_epoch( EEG, markset, [-0.5 1.5], 'newname', 'epochs', 'epochinfo', 'yes');
        EEG = eeg_checkset( EEG );
        EEG = pop_rmbase( EEG, [-500 0] ,[]);
        EEG = eeg_checkset( EEG );
        if event_num == 5% stable should be processed separately
            path = strcat('D:\\datasets\\set',num2str(dataset_num),'\\stable\\EEG\\');
        end
        pop_saveset( EEG, 'filename',strcat('preICA_',char(label(event_num)),'_event.set'),'filepath',path);
    end
end
%% use the function pop_mergeset to merge datasets(according to subjects)
label = {'brake','turn','change','throttle','stable'};
for dataset_num = 1:30
    EEG = pop_loadset(strcat('D:\\datasets\\set',num2str(dataset_num),'\\',char(label(1)),'\\EEG\\preICA_',char(label(1)),'_event.set'));
    for event_num = 2:5
        path = strcat('D:\\datasets\\set',num2str(dataset_num),'\\',char(label(event_num)),'\\EEG\\');
        filename = strcat('preICA_',char(label(event_num)),'_event.set');
        EEG = pop_mergeset(EEG, pop_loadset(strcat(path,'\\',filename)));
        EEG = eeg_checkset( EEG );
    end
    pop_saveset( EEG, 'filename',strcat('preICA_',num2str(dataset_num),'_EEG.set'),'filepath',strcat('D:\\datasets\\set',num2str(dataset_num)));
end

%% use the function pop_mergeset to merge datasets(according to events)
label = {'brake','turn','change','throttle','stable'};
EEG = pop_loadset(strcat('D:\\datasets\\set1\\',char(label(5)),'\\EEG\\preICA_',char(label(5)),'_event.set'));
for dataset_num = 2:30
    path = strcat('D:\\datasets\\set',num2str(dataset_num),'\\',char(label(event_num)),'\\EEG\\');
    filename = strcat('preICA_',char(label(event_num)),'_event.set');
    EEG = pop_mergeset(EEG, pop_loadset(strcat(path,'\\',filename)));
    EEG = eeg_checkset( EEG );
end
pop_saveset( EEG, 'filename',strcat('preICA_stable_EEG.set'),'filepath',strcat('D:\\datasets\\'));
%% merge all the data into one file
EEG = pop_loadset('D:\\datasets\\set1\\preICA_1_EEG_adjusted.set');
for dataset_num = 2:30
    path = strcat('D:\\datasets\\set',num2str(dataset_num),'\\');
    filename = strcat('preICA_',num2str(dataset_num),'_EEG_adjusted.set');
    EEG = pop_mergeset(EEG, pop_loadset(strcat(path,'\\',filename)));
    EEG = eeg_checkset( EEG );
end
pop_saveset( EEG, 'filename','EEGALL','filepath',strcat('D:\\datasets\\'));
%% use EEGLAB to remove bad data manually
eeglab
%% run ICA
EEG = pop_loadset(strcat('D:\\',date,'\\','preICA_event_merged.set'));
EEG = pop_runica(EEG, 'icatype', 'runica', 'extended',1,'pca',59,'interrupt','on');
EEG = eeg_checkset( EEG ); 
EEG = pop_saveset( EEG, 'filename',strcat('postICA_event_merged.set'),'filepath',strcat('D:\\',date,'\\'));
EEG = eeg_checkset( EEG );
%% use EEGLAB to remove artifacts manually
eeglab
%% ECG preprocessing
% data preprocessing & enframing
addpath D:\\eegprocess\\readmultibdfdata
label = {'brake','turn','change','throttle','stable'};
for dataset_num = 1:30
    for event_num = 1:4
        path = strcat('D:\\datasets\\set',num2str(dataset_num),'\\',char(label(event_num)),'\\EEG\\');
        if exist(strcat(path,'merged.set'),'file')
            EEG = pop_loadset(strcat(path,'merged.set'));
        else
            EEG = readbdfdata({'data.bdf' 'evt.bdf'},path);
        end
        EEG.etc.eeglabvers = '2021.1'; % this tracks which version of EEGLAB is being used, you may ignore it
        EEG.setname = 'ECG';
        EEG = eeg_checkset( EEG );
        EEG = pop_chanedit(EEG, 'lookup','D:\\Program Files\\MATLAB\\R2021a\\toolbox\\eeglab2021.1\\plugins\\dipfit\\standard_BEM\\elec\\standard_1005.elc');
        EEG = eeg_checkset( EEG );
        EEG = pop_select( EEG, 'channel',{'ECG'});% select the channel of ECG
        EEG = eeg_checkset( EEG );
        EEG = pop_eegfiltnew(EEG, 'locutoff',0.01,'hicutoff',200,'plotfreqz',0);
        EEG = eeg_checkset( EEG );
        % EEG = pop_resample( EEG, 200);
        % EEG = eeg_checkset( EEG ); 
        pop_saveset( EEG, 'filename',strcat(char(label(event_num)),'_ECG_postconpro'),'filepath',path);
    end
end
addpath D:\\
label = {'brake','turn','change','throttle','stable'};
for dataset_num = 1:30
    for event_num = 1:5
        path = strcat('D:\\datasets\\set',num2str(dataset_num),'\\',char(label(event_num)),'\\EEG\\');
        if event_num == 5
            path = strcat('D:\\datasets\\set',num2str(dataset_num),'\\turn\\EEG\\');
            EEG = pop_loadset(strcat(path,'\\turn_ECG_postconpro.set'));
        else
            EEG = pop_loadset(strcat(path,'\\',char(label(event_num)),'_ECG_postconpro.set'));
        end
        EEG.etc.eeglabvers = '2021.1'; % this tracks which version of EEGLAB is being used, you may ignore it
        EEG.setname = 'ECG';
        EEG = eeg_checkset( EEG );
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
        EEG = pop_epoch( EEG, markset, [-0.5 1.5], 'newname', 'epochs', 'epochinfo', 'yes');
        EEG = eeg_checkset( EEG );
        EEG = pop_rmbase( EEG, [-500 0] ,[]);
        EEG = eeg_checkset( EEG );
        if event_num == 5
            path = strcat('D:\\datasets\\set',num2str(dataset_num),'\\stable\\EEG\\');
        end
        pop_saveset( EEG, 'filename',strcat('preICA_',char(label(event_num)),'_ECG_event.set'),'filepath',path);
    end
end
label = {'brake','turn','change','throttle','stable'};
for dataset_num = 1:30
    EEG = pop_loadset(strcat('D:\\datasets\\set',num2str(dataset_num),'\\',char(label(1)),'\\EEG\\preICA_',char(label(1)),'_ECG_event.set'));
    for event_num = 2:5
        path = strcat('D:\\datasets\\set',num2str(dataset_num),'\\',char(label(event_num)),'\\EEG\\');
        filename = strcat('preICA_',char(label(event_num)),'_ECG_event.set');
        EEG = pop_mergeset(EEG, pop_loadset(strcat(path,'\\',filename)));
        EEG = eeg_checkset( EEG );
    end
    pop_saveset( EEG, 'filename',strcat('preICA_',num2str(dataset_num),'_ECG.set'),'filepath',strcat('D:\\datasets\\set',num2str(dataset_num)));
end