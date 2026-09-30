classdef TemporalAttentionLayer < nnet.layer.Layer & nnet.layer.Formattable
    properties (Learnable)
        Weights 
        Bias
    end
    methods
        function layer = TemporalAttentionLayer(name)
            layer.Name = name;
            layer.Description = "1D Temporal Attention (CBAM)";
            
            % Explicitly initialize the weights with the 'SCU' format
            % S = Spatial (Time), C = Channels (Max & Mean = 2), U = Unspecified (Filters = 1)
            layer.Weights = dlarray(randn([7, 2, 1], 'single') * sqrt(2/14), 'SCU'); 
            layer.Bias = dlarray(zeros([1, 1, 1], 'single'), 'U');
        end
        function Z = predict(layer, X)
            % 1. Dynamically find which dimension (1, 2, or 3) corresponds to Channels ('C')
            cDim = finddim(X, 'C');
            if isempty(cDim)
                cDim = 1; % Failsafe in case of an unformatted test
            end
            
            % 2. Perform Pooling exactly on this dimension. 
            avg_pool = mean(X, cDim);
            max_pool = max(X, [], cDim);
            
            % 3. Concatenation along the 'C' dimension
            concat_dl = cat(cDim, avg_pool, max_pool);
            
            % 4. 1D Convolution. 
            conv_out = dlconv(concat_dl, layer.Weights, layer.Bias, 'Padding', 'same');
            
            % 5. Mask Creation (Temporal Re-weighting)
            attention_mask = sigmoid(conv_out);
            
            % 6. Apply the mask to the original signal
            Z = X .* attention_mask;
        end
    end
end