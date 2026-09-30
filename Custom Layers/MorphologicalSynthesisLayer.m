classdef MorphologicalSynthesisLayer < nnet.layer.Layer & nnet.layer.Formattable
    methods
        function layer = MorphologicalSynthesisLayer(name)
            layer.Name = name;
            layer.Description = "Cross-Scale Morphological Fusion (Max/Min)";
        end
        function Z = predict(~, X)
            % 1. Strip the formatted tags to access the underlying unformatted data array
            if isa(X, 'dlarray')
                fmt = dims(X);
                X_data = stripdims(X);
            else
                fmt = '';
                X_data = X;
            end
            sz = size(X_data);
            
            % Dynamically find WHICH dimension has a size of 5
            % This approach ensures it won't fail regardless of input format.
            cDim = find(sz == 5, 1);
            if isempty(cDim)
                error("Dimension error! Expected 5 Wavelet levels. Got: %s", mat2str(sz));
            end
            
            % Create dynamic indices to extract exactly the channels we want
            idx = repmat({':'}, 1, ndims(X_data));
            
            % Safely extract the 3 desired levels
            idx{cDim} = 4; W_winner = X_data(idx{:});
            idx{cDim} = 5; W_second = X_data(idx{:});
            idx{cDim} = 3; W_third  = X_data(idx{:});
            
            % Morphological Synthesis (Cross-Scale Fusion)
            C_peak = max(W_winner, W_second);
            C_valley = min(W_winner, W_third);
            
            % Concatenate ALONG THE SAME DIMENSION (so the output size remains 5)
            Z_data = cat(cDim, W_winner, W_second, W_third, C_peak, C_valley);
            
            % Restore the original dlarray format for the next layer
            if ~isempty(fmt)
                Z = dlarray(Z_data, fmt);
            else
                Z = Z_data;
            end
        end
    end
end