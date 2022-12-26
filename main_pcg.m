clear all;
close all;
clc;

prompt = 'Enter .wav filename (without .extension):'; 
%% example: "/home/mertcokelek/Desktop/Dataset/2530_TV";
audioname = input(prompt, 's');
wav_path = strcat(audioname, ".wav");
output_path = strcat(audioname, "_saliency.mat");

disp(["Processing: " wav_path]);
disp(["Output Path: " output_path]);

infoaudio = audioinfo(wav_path);
sdata = audioread(wav_path);
t = linspace(0, infoaudio.Duration, length(sdata));
Fs = infoaudio.SampleRate;
audio_in_seconds = zeros(int16(infoaudio.Duration), Fs, 1);
disp(Fs);

%% I guess it's processing 1-sec clips. (Short-time).
for i = 1:infoaudio.Duration;
    currentClip = reshape(sdata((i-1)*Fs+1: i*Fs, :), Fs, 1);
    W = currentClip(:, 1);
       
    sW = spectrum1DwithMelCepstrumTrial(W, infoaudio,1)';
    windowSize = infoaudio.SampleRate/2;
    b = (1/windowSize)*ones(1,windowSize);
    a = 1;

    hW = filter(b, a, sW);
    audio_in_seconds(i, :, 1) = hW;
end

save(output_path, 'saliency', '-v6');