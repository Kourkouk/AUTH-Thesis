classdef WaveletChannelAttentionLayer < nnet.layer.Layer & nnet.layer.Formattable
    properties (Learnable)
        Weights1
        Bias1
        Weights2
        Bias2
    end
    
    properties
        NumChannels
        ReductionRatio
    end
    
    methods
        function layer = WaveletChannelAttentionLayer(numChannels, reductionRatio, name)
            layer.Name = name;
            layer.Description = "1D SE Channel Attention";
            layer.NumChannels = numChannels;
            
            % Set default reduction ratio if not provided
            if nargin < 2 || isempty(reductionRatio)
                reductionRatio = 2;
            end
            layer.ReductionRatio = reductionRatio;
            
            % Calculate the hidden dimension size (bottleneck design)
            hiddenDim = max(1, round(numChannels / reductionRatio));
            
            % Initialize Learnable Parameters for the Multi-Layer Perceptron (MLP)
            layer.Weights1 = randn([hiddenDim, numChannels], 'single') * sqrt(2 / numChannels);
            layer.Bias1 = zeros([hiddenDim, 1], 'single');
            
            layer.Weights2 = randn([numChannels, hiddenDim], 'single') * sqrt(2 / hiddenDim);
            layer.Bias2 = zeros([numChannels, 1], 'single');
        end
        
        function Z = predict(layer, X)
            % 1. Strip the formatted tags to access the underlying unformatted data array
            if isa(X, 'dlarray')
                fmt = dims(X);
                X_data = stripdims(X);
            else
                fmt = '';
                X_data = X;
            end
            
            C = size(X_data, 1);
            
            % 2. Squeeze Operation: Global Average Pooling across the time/spatial dimension (dimension 2)
            s = mean(X_data, 2); 
            s_flat = reshape(s, C, []); 
            
            % 3. Excitation Operation: Two-layer MLP with bottleneck
            % First layer with ReLU activation (dimensionality reduction)
            h = relu(layer.Weights1 * s_flat + layer.Bias1);
            
            % Second layer with Sigmoid activation (dimensionality restoration & attention score generation)
            a_flat = sigmoid(layer.Weights2 * h + layer.Bias2); 
            
            % 4. Reshape the attention mask to broadcast correctly across the original dimensions
            szA = size(X_data);
            szA(2) = 1; 
            
            a = reshape(a_flat, szA);
            
            % 5. Scale/Reweight the original input features with the computed attention mask
            Z_data = X_data .* a;
            
            % 6. Restore the original dlarray format for the next layer
            if ~isempty(fmt)
                Z = dlarray(Z_data, fmt);
            else
                Z = Z_data;
            end
        end
    end
end