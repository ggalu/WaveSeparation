function [wave_forward, wave_backward, t_pad] = separate_waves_nodisp(sg1, sg2, fs, x1, x2, x_target, c0)
    
    % This function operates a wave deconvolution in the Fourier domain,
    % decomposing the Input wave into its two components: incident and
    % reflected. This algorithm needs the Input wave acquisitions sg1 and sg2 
    % in 2 different locations x1 and x2; the sampling frequency fs; the
    % speed of sound c0, and the location at which calculate the decomposed
    % signals; 

    dt=1/fs;
    N = length(sg1);
    
    % 1. Zero Padding (Mandatory for FFT/IFFT resolution and stability)
    N_fft = 2^nextpow2(N * 3); 
    sg1_pad = [sg1; zeros(N_fft - N, 1)];
    sg2_pad = [sg2; zeros(N_fft - N, 1)];
    t_pad = (0:N_fft-1)' * dt;
    
    % 2. Angular frequency vector (w)
    f = fs * (0:N_fft-1)' / N_fft;
    w = 2 * pi * f;
    % Map frequencies beyond Nyquist limit as negative for a real IFFT
    nyquist_idx = floor(N_fft/2) + 1;
    w(nyquist_idx+1:end) = w(nyquist_idx+1:end) - 2*pi*fs;
    
    % 3. Wavenumber (k) IS PURELY LINEAR (No dispersion assumed)
    k = w ./ c0;
    
    % 4. Regularization: necessary to avoid division by zero at frequencies 
    % where the distance between SGs is a multiple of the wavelength.
    alpha = 0.05 / abs(x2 - x1); 
    gamma = alpha + 1i * k; 
    
    % 5. FFT of the padded signals
    E1 = fft(sg1_pad);
    E2 = fft(sg2_pad);
    
    dx = x2 - x1; 
    den = exp(gamma * dx) - exp(-gamma * dx);
    
    % 6. Wave separation at the location of SG1 (x1)
    A_x1 = (E1 .* exp(gamma * dx) - E2) ./ den; % Forward traveling wave
    B_x1 = (E2 - E1 .* exp(-gamma * dx)) ./ den; % Backward traveling wave
    
    % 7. Spatial shift (phase-shift) to the specimen interface (x_target)
    dist = x_target - x1;
    A_target = A_x1 .* exp(-gamma * dist);
    B_target = B_x1 .* exp(+gamma * dist);
    
    % 8. IFFT (Return to time domain, keeping only the real part)
    wave_forward = real(ifft(A_target));
    wave_backward = real(ifft(B_target));
end