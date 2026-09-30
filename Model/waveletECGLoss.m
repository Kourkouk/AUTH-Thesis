function loss = waveletECGLoss(y, t, lambda)
    % Ensure inputs are real numbers
    y = real(y);
    t = real(t);
    
    % Loss 1: Time-domain MSE
    lossECG = mean((y-t).^2, 'all');
    
    % Perform MODWT on both predicted and target signals
    waveletY = real(dlmodwt(y, [], [], 5));
    waveletT = real(dlmodwt(t, [], [], 5));
    
    % Extract only the 5 detail levels
    waveletY = waveletY(1:5, :, :, :);
    waveletT = waveletT(1:5, :, :, :);
    
    % Loss 2: Frequency/Wavelet-domain MSE
    lossWavelet = mean((waveletY - waveletT).^2, 'all');
    
    % Total Multi-Objective Loss
    loss = lossECG + lambda * lossWavelet;
end