## pcg-saliency
1D MCSR-based saliency detection on PCG signals.

### Dependencies
- MATLAB (Tested on R2022b)
- Python (Tested on 3.8.8)

### Requirements
- scipy
- librosa
- numpy
- matplotlib

### Run
**Saliency Detection with MATLAB:** You may also run with the MATLAB GUI. 

Here, terminal command is given:

```./matlab -nodisplay -nosplash -nodesktop -r "run('path/to/repo/pcg-saliency/main_pcg.m');exit;"```

Example prompt (assume that ```/home/mertcokelek/Desktop/PCG-Data/``` folder contains 2530_AV.wav, 2530_TV.wav, etc.):



```>> Enter .wav filename (without .extension): /home/mertcokelek/Desktop/PCG-Data/2530_AV```

It will produce ```/home/mertcokelek/Desktop/PCG-Data/2530_AV_saliency.mat```

Then, you may see the **jupyter notebook** to visualize your audio waveforms and saliency curves.
