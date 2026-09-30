classdef RadarECGWaveletCrossAttentionLayer < nnet.layer.Layer & nnet.layer.Formattable
    properties (Learnable)
        Query
        KeyWeights
        KeyBias
        ChannelEmbedding
        ValueWeights
        ValueBias
    end
    
    properties
        NumRadarChannels
        NumECGWavelets
        KeyDimension
        WindowRadius
    end
    
    methods
        function layer = RadarECGWaveletCrossAttentionLayer(numRadarChannels, numECGWavelets, keyDimension, windowRadius, name)
            layer.Name = name;
            layer.Description = "Local Radar-to-ECG Wavelet Cross-Attention (Vectorized & Stable)";
            layer.NumRadarChannels = numRadarChannels;
            layer.NumECGWavelets = numECGWavelets;
            layer.KeyDimension = keyDimension;
            layer.WindowRadius = windowRadius;
            
            % Learnable Parameters initialization
            layer.Query = randn(numECGWavelets, keyDimension, 'single') * sqrt(2/keyDimension);
            layer.KeyWeights = randn(keyDimension, 1, 'single') * sqrt(2);
            layer.KeyBias = zeros(keyDimension, 1, 'single');
            layer.ChannelEmbedding = randn(keyDimension, numRadarChannels, 'single') * 0.05;
            layer.ValueWeights = randn(1, 1, 'single') * 0.1;
            layer.ValueBias = zeros(1, 1, 'single');
        end
        
        function Z = predict(layer, X)
            if isa(X, 'dlarray')
                fmt = dims(X);
                X = stripdims(X);
            else
                fmt = '';
            end
             
            orig_sz = size(X);
            C = orig_sz(1);
            T = orig_sz(2);
            
            if C ~= layer.NumRadarChannels
                error("Expected %d radar wavelet channels, received %d.", layer.NumRadarChannels, C);
            end
            
            % "Flatten" all Batch dimensions (3, 4, etc.) into ONE.
            X_3D = reshape(X, C, T, []);
            B = size(X_3D, 3); % B contains ALL batch elements, without losing anything
            
            R = layer.WindowRadius;
            K = layer.KeyDimension;
            E = layer.NumECGWavelets;
            W = 2 * R + 1; % Number of offsets per channel
            N = C * W;     % Total tokens
            
            % =========================================================
            % 1. VECTORIZED WINDOWING
            % =========================================================
            Zpad = zeros(C, R, B, 'like', X_3D);
            X_pad = cat(2, Zpad, X_3D, Zpad);
            
            X_windows = zeros(C, W, T, B, 'like', X_3D);
            
            token_idx = 1;
            for offset = -R:R
                start_idx = R + 1 - offset;
                window_slice = X_pad(:, start_idx : start_idx+T-1, :);
                
                X_windows(:, token_idx, :, :) = reshape(window_slice, [C, 1, T, B]);
                token_idx = token_idx + 1;
            end
            
            X_windows_perm = permute(X_windows, [2, 1, 3, 4]); 
            X_flat = reshape(X_windows_perm, N, T, B);
            
            % =========================================================
            % 2. COMPUTE VALUES & KEYS 
            % =========================================================
            values = layer.ValueWeights(1,1) .* X_flat + layer.ValueBias(1,1);
            
            keys_base = reshape(layer.KeyWeights, [K, 1, 1, 1]) .* reshape(X_flat, [1, N, T, B]);
            CE = repelem(layer.ChannelEmbedding, 1, W); 
            
            keys = keys_base + reshape(layer.KeyBias, [K, 1, 1, 1]) + reshape(CE, [K, N, 1, 1]);
            
            % =========================================================
            % 3. COMPUTE ATTENTION SCORES
            % =========================================================
            keys_mat = reshape(keys, K, []); 
            scores_mat = (layer.Query * keys_mat) ./ sqrt(single(K));
            scores = reshape(scores_mat, [E, N, T, B]);
            
            % Softmax (Manual computation to avoid overflow)
            maxScores = max(scores, [], 2); 
            expScores = exp(scores - maxScores);
            attention = expScores ./ sum(expScores, 2); 
            
            % =========================================================
            % 4. WEIGHTED SUM & RESIDUAL CONNECTION
            % =========================================================
            weighted_values = attention .* reshape(values, [1, N, T, B]);
            out = sum(weighted_values, 2); 
            
            Z_3D = reshape(out, [E, T, B]) + X_3D;
            
            % Restore: Rebuild the array
            out_sz = orig_sz;
            out_sz(1) = E;
            Z = reshape(Z_3D, out_sz);
            
            if ~isempty(fmt)
                Z = dlarray(Z, fmt);
            end
        end
    end
end