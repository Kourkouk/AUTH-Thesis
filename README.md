# Contactless ECG Reconstruction from CW Radar Signals

## Overview
This project focuses on the development and training of deep learning architectures that learn to autonomously reconstruct continuous Electrocardiogram (ECG) waveforms from non-contact Continuous-Wave (CW) radar measurements. The simulation is built upon clinical data from 30 healthy volunteers, recorded under various physiological conditions (Resting, Apnea, Valsalva, Tilt Up/Down). 

The solution leverages a combination of Multiresolution Analysis (Wavelets) for signal purification and 1-Dimensional Convolutional Neural Networks (1D-CNNs) for cross-domain translation, operating purely as an experimental research endeavor.

## Input Data and Target Signals
* **Inputs (Radar):** The model relies on raw mechanical chest displacement data extracted from CW radar In-Phase and Quadrature (I/Q) signals. The phase is unwrapped, downsampled to 200 Hz, and segmented into 2048-point arrays (representing 10.24 seconds of temporal context).
* **Targets (ECG):** The ground-truth electrical signals are carefully bandpass-filtered (0.5 Hz - 40 Hz) to eliminate baseline wander and high-frequency noise, providing a morphologically pristine target for the neural network.
* **Challenge:** The network must learn the highly non-linear mapping between mechanical chest micro-motions and electrical cardiac activity while actively suppressing overwhelming low-frequency artifacts caused by human respiration.

## Architecture
The reconstruction system is structurally divided into two main experimental subsystems:
1. **Feature Extraction (Multiresolution Analysis) - MODWT & 1D-CNN:** The raw signal is processed through a Maximal Overlap Discrete Wavelet Transform (MODWT) using a `sym4` wavelet to decompose the frequencies. A custom 1D-CNN acts as the primary encoder-decoder, using aggressive stride convolutions and Transposed 1D-CNNs to compress and reconstruct the waveform while filtering out the respiratory baseline.
2. **Context & Morphology (Brain) - BiLSTM & Custom Attention:** The temporal rhythm (Inter-Beat Interval) is modeled using a Bidirectional LSTM. To combat morphological flattening (loss of sharp R-peaks), the architecture is modular, allowing specific custom layers to be swapped in for ablation studies:
   * **Wavelet Channel Attention (1D-SE):** Dynamically re-weights frequency channels.
   * **Temporal Attention (1D-CBAM):** Evaluates local variances to explicitly maximize SNR around physiological peaks.
   * **Radar-to-ECG Cross Attention:** Attempts localized cross-domain translation.
   * **Morphological Synthesis:** A zero-parameter, physics-informed fusion using max/min operations on dominant wavelet scales.

## Training & Results
Through extensive hyperparameter sweeps and structural modifications, the models were trained using a custom dual-objective loss function (`waveletECGLoss`). This function computes the Mean Squared Error (MSE) simultaneously in the time domain and the wavelet domain to heavily penalize over-smoothing.

The system was evaluated strictly against stationary and non-stationary clinical scenarios. The integration of the physics-informed `MorphologicalSynthesisLayer` achieved the highest quantitative performance, dropping the Mean Test MSE by **~13.4%**. However, the experimental findings confirm a critical physical limitation: optimizing for global regression metrics effectively suppresses inter-beat noise but fails to fully synthesize the sharp, high-frequency transients of the QRS complex.

## Technologies & Libraries Used
* **MATLAB** (R2026a)
* **Deep Learning Toolbox** (for building and training the custom CNN, BiLSTM, and Attention layers)
* **Wavelet Toolbox** (for the MODWT integration and multiresolution analysis)
* **Signal Processing Toolbox** (for custom data initialization and bandpass filtering)

## Project Structure
* `docs/`: Directory containing the full documentation.
  * `AUTH_Thesis_Template__English_.pdf`: The complete thesis document detailing the mathematical formulations and experimental results.
* `data_preprocessing/`: Scripts for dynamic data loading (`signalDatastore`), scenario filtering, and temporal segmentation.
* `custom_layers/`: Custom neural network classes (`WaveletChannelAttentionLayer`, `TemporalAttentionLayer`, `RadarECGWaveletCrossAttentionLayer`, `MorphologicalSynthesisLayer`).
* `models_and_training/`: The core network compilation and the training loop utilizing the custom `waveletECGLoss`.
* `utils/`: Rendering and visualization scripts (`helperPlotData`) to plot the mechanical radar signals alongside the measured and reconstructed ECGs.
