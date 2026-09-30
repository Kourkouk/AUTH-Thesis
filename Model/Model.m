% The code is saperated into tow versions, a version where 
% only the 3/5 scenarios are being loaded ("Apnea", "Resting", "Valsalva")
% and a version whith all of the 5/5 scenarios ("Apnea", "Resting",
% "Valsalva", "TiltUp", "TiltDown").

% ============================================================
% 1. CLEAN DATA LOADING ("Apnea", "Resting", "Valsalva")
% ============================================================
outputBaseDir = 'C:\Users\Kourkouk\Documents\ProcessedDataset';
radarTrainValDs = signalDatastore(fullfile(outputBaseDir,"trainVal","radar"));
ecgTrainValDs   = signalDatastore(fullfile(outputBaseDir,"trainVal","ecg"));
radarTestDs     = signalDatastore(fullfile(outputBaseDir,"test","radar"));
ecgTestDs       = signalDatastore(fullfile(outputBaseDir,"test","ecg"));

trainCats = filenames2labels(radarTrainValDs,'ExtractBefore','_radar');
testCats  = filenames2labels(radarTestDs,'ExtractBefore','_radar');

% -- Here we ignore the word 'Tilt' because we only loading the 3 first scenarios --
tiltIdxTrain = contains(string(trainCats), 'Tilt', 'IgnoreCase', true);
radarTrainValDs = subset(radarTrainValDs, ~tiltIdxTrain);
ecgTrainValDs   = subset(ecgTrainValDs, ~tiltIdxTrain);
trainCats(tiltIdxTrain) = [];

tiltIdxTest = contains(string(testCats), 'Tilt', 'IgnoreCase', true);
radarTestDs = subset(radarTestDs, ~tiltIdxTest);
ecgTestDs   = subset(ecgTestDs, ~tiltIdxTest);
testCats(tiltIdxTest) = [];
% -----------------------------------------------------------------------

ecgTrainValDs = transform(ecgTrainValDs, @helperNormalize);
ecgTestDs     = transform(ecgTestDs, @helperNormalize);
trainValDs = combine(radarTrainValDs, ecgTrainValDs);
testDs     = combine(radarTestDs, ecgTestDs);

% % ============================================================
% % 1. CLEAN DATA LOADING ("Apnea", "Resting", "Valsalva", "TiltUp", "TiltDown")
% % ============================================================
% outputBaseDir = 'C:\Users\Kourkouk\Documents\ProcessedDataset';
% 
% radarTrainValDs = signalDatastore(fullfile(outputBaseDir,"trainVal","radar"));
% ecgTrainValDs   = signalDatastore(fullfile(outputBaseDir,"trainVal","ecg"));
% radarTestDs     = signalDatastore(fullfile(outputBaseDir,"test","radar"));
% ecgTestDs       = signalDatastore(fullfile(outputBaseDir,"test","ecg"));
% 
% trainCats = filenames2labels(radarTrainValDs,'ExtractBefore','_radar');
% testCats  = filenames2labels(radarTestDs,'ExtractBefore','_radar');
% 
% ecgTrainValDs = transform(ecgTrainValDs, @helperNormalize);
% ecgTestDs     = transform(ecgTestDs, @helperNormalize);
% 
% trainValDs = combine(radarTrainValDs, ecgTrainValDs);
% testDs     = combine(radarTestDs, ecgTestDs);

splitIndices = splitlabels(trainCats, 0.8);
trainDs = subset(trainValDs, splitIndices{1});
valDs   = subset(trainValDs, splitIndices{2});

disp('Loading data in memory...');
trainData = readall(trainDs);
valData   = readall(valDs);
testData  = readall(testDs);

% ============================================================
% 2. MODEL ARCHITECTURE (Attention)
% ============================================================
layers_attention = [
    sequenceInputLayer(1, MinLength=2048, Name="input")

    modwtLayer('Level', 5, 'IncludeLowpass', false, 'SelectedLevels', 1:5, "Wavelet", "sym4", Name="modwt")

    %RadarECGWaveletCrossAttentionLayer(5, 5, 16, 3, "cross_att")
    
    WaveletChannelAttentionLayer(5, 2, "wavelet_attention")

    %TemporalAttentionLayer("temporal_att")

    %MorphologicalSynthesisLayer("morph_synthesis")
    
    flattenLayer(Name="flatten")

    layerNormalizationLayer(Name="ln")
    dropoutLayer(0.3, Name="drop1")

    convolution1dLayer(64, 8, Padding="same", Stride=8, Name="conv1")
    reluLayer(Name="relu1")
    batchNormalizationLayer(Name="bn1")
    dropoutLayer(0.3, Name="drop2")

    maxPooling1dLayer(2, Padding="same", Name="pool1")

    convolution1dLayer(32, 8, Padding="same", Stride=4, Name="conv2")
    reluLayer(Name="relu2")
    batchNormalizationLayer(Name="bn2")

    maxPooling1dLayer(2, Padding="same", Name="pool2")

    transposedConv1dLayer(32, 8, Cropping="same", Stride=4, Name="deconv1")
    reluLayer(Name="relu3")

    transposedConv1dLayer(64, 8, Cropping="same", Stride=8, Name="deconv2")

    bilstmLayer(8, OutputMode="sequence", Name="bilstm")

    fullyConnectedLayer(4, Name="fc1")
    fullyConnectedLayer(2, Name="fc2")
    fullyConnectedLayer(1, Name="fc_out")
];
lambda = 0.4; 
options = trainingOptions("adam",...
    MaxEpochs=200, MiniBatchSize=450, InitialLearnRate=0.001,...
    ValidationData={valData(:,1),valData(:,2)},...
    ValidationFrequency=40, Verbose=1, Shuffle="every-epoch",...
    OutputNetwork="best-validation", Plots="training-progress",...
    InputDataFormats="CTB", TargetDataFormats="CTB",...
    ExecutionEnvironment="parallel"); 
lossFcn = @(y,t) waveletECGLoss(y,t,lambda);

% ============================================================
% 3. TRAINING & EVALUATION
% ============================================================
disp('=== Training: Wavelet Attention ===');
[net_attention, ~] = trainnet(trainData(:,1), trainData(:,2), layers_attention, lossFcn, options);

disp('=== Testing: Model Evaluation ===');
ecgTestRe_att = minibatchpredict(net_attention, testData(:,1), InputDataFormats="CTB", ExecutionEnvironment="cpu");
ecgTestReCell_att = squeeze(mat2cell(ecgTestRe_att, 1, 2048, ones(size(testData,1),1)));
lossAttModel = cellfun(@mse, ecgTestReCell_att, testData(:,2));
disp(['Mean Test MSE: ', num2str(mean(lossAttModel))]);

figure('Name', 'MSE Distribution');
histogram(lossAttModel, 'Normalization', 'probability', 'BinWidth', 0.002, 'FaceColor', [0 0.4470 0.7410], 'FaceAlpha', 0.7);
ylabel("Probability"); xlabel("Test MSE");
title("Performance: Wavelet Attention Model"); grid on;

% ============================================================
% 4. VISUALIZE RANDOM SAMPLES
% ============================================================
disp('=== Generating 5 Random Sample Plots ===');

% Customized for 5 scenarios
%targetCategories = ["Apnea", "Resting", "Valsalva", "TiltUp", "TiltDown"];

% Customized for 3 scenarios
targetCategories = ["Apnea", "Resting", "Valsalva"];

selectedIndices = zeros(1, length(targetCategories));

for i = 1:length(targetCategories)
    currentCat = targetCategories(i);
    catIndices = find(testCats == currentCat);
    if ~isempty(catIndices)
        selectedIndices(i) = catIndices(randi(length(catIndices)));
    end
end

% Cleanup of any records that were not found (e.g., if a patient had no entry)
selectedIndices(selectedIndices == 0) = [];

if ~isempty(selectedIndices)
    figure('Name', 'Reconstruction Evaluation (3 Scenarios)', 'Color', [0.15 0.15 0.15]);
    helperPlotData(testDs, selectedIndices, net_attention);
else
    disp('Records not found for Graphs');
end