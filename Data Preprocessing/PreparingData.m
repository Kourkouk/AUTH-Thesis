% =========================================================================
% Script: DataPreparation.m
% Description: Segmentation of Raw Data (Subjects 11–20) with ECG Filtering
% =========================================================================
inputBaseDir = 'C:\Users\...\Documents\MyRawData'; 
outputBaseDir = 'C:\Users\...\Documents\ProcessedDataset'; 

segmentLength = 2048; 
downsampleFactor = 10; 

foldersToMake = {fullfile(outputBaseDir, 'trainVal', 'radar'), ...
                 fullfile(outputBaseDir, 'trainVal', 'ecg'), ...
                 fullfile(outputBaseDir, 'test', 'radar'), ...
                 fullfile(outputBaseDir, 'test', 'ecg')};
for i = 1:length(foldersToMake)
    if ~exist(foldersToMake{i}, 'dir'), mkdir(foldersToMake{i}); end
end

scenarios = ["Resting", "Valsalva", "Apnea", "TiltUp", "TiltDown"];
disp('Data processing begins (Subjects 11-20). ');

for subject = 11:20
    if subject <= 18
        destFolder = "trainVal";
    else
        destFolder = "test";
    end
    
    subjName = sprintf('GDN%04d', subject);
    subjPath = fullfile(inputBaseDir, subjName);
    
    if ~exist(subjPath, 'dir')
        warning(['File not found: ', subjName]);
        continue; {}
    end
    
    files = dir(fullfile(subjPath, '*.mat'));
    
    for f = 1:length(files)
        fileName = files(f).name;
        
        containsScenario = false;
        currentScenario = "";
        for s = scenarios
            if contains(fileName, s, 'IgnoreCase', true)
                containsScenario = true;
                currentScenario = s;
                break;
            end
        end
        
        if ~containsScenario, continue; end
        
        disp(['Processing: ', fileName]);
        data = load(fullfile(subjPath, fileName));
        
        % ECG Cleanup
        ecg_raw = data.tfm_ecg1;
        ecg_raw(~isfinite(ecg_raw)) = NaN; 
        ecg_raw = fillmissing(ecg_raw, 'linear'); 
        ecg_raw(isnan(ecg_raw)) = 0; 
        
        % Cleanup & Radar Phase
        radar_i_raw = data.radar_i;
        radar_q_raw = data.radar_q;
        radar_i_raw(~isfinite(radar_i_raw)) = NaN;
        radar_i_raw = fillmissing(radar_i_raw, 'linear');
        radar_i_raw(isnan(radar_i_raw)) = 0;
        
        radar_q_raw(~isfinite(radar_q_raw)) = NaN;
        radar_q_raw = fillmissing(radar_q_raw, 'linear');
        radar_q_raw(isnan(radar_q_raw)) = 0;
        
        radar_complex = radar_i_raw + 1i * radar_q_raw;
        radar_raw = unwrap(angle(radar_complex));
        
        % Downsampling to 200Hz 
        ecg_down = decimate(ecg_raw, downsampleFactor);
        radar_down = decimate(radar_raw, downsampleFactor);
        
        % Application of a bandpass filter to the ECG (0.5 Hz to 40 Hz)
        % Cuts out wandering (low) and mains/muscle noise (high)
        ecg_down = bandpass(ecg_down, [0.5, 40], 200);
        
        % Segmentation to 2048
        numSegments = floor(length(ecg_down) / segmentLength);
        
        for k = 1:numSegments
            idx_start = (k-1)*segmentLength + 1;
            idx_end = k*segmentLength;
            
            ecg_segment = ecg_down(idx_start:idx_end);
            radar_segment = radar_down(idx_start:idx_end);
            
            baseName = sprintf('%s_%%s_S%02d_%04d.mat', currentScenario, subject, k);
            radarName = sprintf(baseName, 'radar');
            ecgName = sprintf(baseName, 'ecg');
            
            radarSavePath = fullfile(outputBaseDir, destFolder, 'radar', radarName);
            ecgSavePath = fullfile(outputBaseDir, destFolder, 'ecg', ecgName);
            
            dataEcg = reshape(ecg_segment, 1, segmentLength);
            save(ecgSavePath, 'dataEcg'); 
            
            dataRadar = reshape(radar_segment, 1, segmentLength);
            save(radarSavePath, 'dataRadar');
        end
    end
end
disp('Completed successfully!');
