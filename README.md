Contactless ECG Reconstruction from CW Radar Signals

Author: Konstantinos Kourkoutmanos

Institution: Aristotle University of Thessaloniki, School of Informatics

Supervising Professor: Nikolaos Laskaris

This repository contains the codebase and the full documentation for my undergraduate thesis, "Contactless ECG Reconstruction from CW Radar Signals using Morphology-Aware Architectures and Attention-Guided Neural Networks".

📄 Read the full thesis here: AUTH_Thesis_Template__English_.pdf

🔬 About the Project (Experimental Research)

This project is primarily an experimental and research-focused endeavor. The main objective is to investigate the feasibility of reconstructing continuous Electrocardiogram (ECG) waveforms from non-contact Continuous-Wave (CW) radar measurements of mechanical chest displacement using Deep Learning.

A key challenge addressed in this research is the presence of dominant respiratory and low-frequency motion artifacts, which obscure the subtle mechanical manifestations of cardiac activity. The codebase reflects the evolutionary methodology of the research, progressing from a baseline 1D-CNN toward advanced, custom-built wavelet- and attention-based architectures.

🧩 The Modular Architecture & Custom Layers

It is important to note that not all custom layers are used simultaneously in a single production model. Instead, the codebase is modular. The custom neural network layers were developed to test specific hypotheses and overcome distinct morphological challenges encountered during the research (as detailed in Chapter 6 of the thesis).

Depending on the specific experiment being run, different layers are swapped into the core architecture:

WaveletChannelAttentionLayer (Squeeze-and-Excitation): Used to dynamically re-weight the MODWT multi-scale representation to suppress out-of-band non-stationary noise.

TemporalAttentionLayer (1D CBAM): An experimental layer designed to test whether a dynamic temporal mask could explicitly maximize local SNR and sharpen R-peaks.

RadarECGWaveletCrossAttentionLayer: A Local Cross-Attention mechanism attempting cross-domain translation (Radar Keys to ECG Queries) to bridge morphological gaps.

MorphologicalSynthesisLayer: A zero-parameter, physics-informed layer that performs hard pruning and non-linear morphological operations (max/min) on dominant wavelet scales.

📁 Repository Structure

docs/ (or root): Contains the full thesis document (AUTH_Thesis_Template__English_.pdf).

data_preprocessing/: MATLAB scripts for cleaning raw .mat files, phase demodulation, ground-truth bandpass filtering, and generating 2048-point sequences.

custom_layers/: The experimental deep learning layers mentioned above, built natively in MATLAB using nnet.layer.Layer.

models_and_training/: Scripts defining the core 1D-CNN + BiLSTM architecture, the custom dual-domain spectral loss function (waveletECGLoss), and the training loops.

utils/: Visualization and evaluation functions used to assess Mean Squared Error (MSE) and morphological fidelity across different physiological scenarios (Resting, Apnea, Valsalva, etc.).

⚙️ Requirements

MATLAB R2026a (or newer)

Deep Learning Toolbox

Wavelet Toolbox (for MODWT integration)

📊 Key Findings

While advanced attention mechanisms and morphological fusion significantly reduced global Mean Squared Error (MSE) and successfully suppressed inter-beat noise, the research highlights a critical discrepancy: optimizing for global regression metrics does not guarantee the successful synthesis of sharp, high-frequency clinical transients (like the QRS complex). This repository serves as a foundational exploration into the physical and architectural limits of mapping macroscopic mechanical vibrations to micro-transient electrical signals.
