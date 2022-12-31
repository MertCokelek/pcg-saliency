clear all;
close all;
clc;

%%
proj_dir = '/home/tanjary21/Desktop/bio/project/';
data_dir = strcat(proj_dir, '/', 'data/moodyData/physionet.org/files/circor-heart-sound/1.0.3/training_data'); % in this folder there should be .wav files
output_dir = strcat(proj_dir, '/', 'output');

files = dir(data_dir);

for file_ix = 1:30 %length(files)
    file_name = files(file_ix).name;
    if contains(file_name, '.wav')
        saliency = forward(data_dir, file_name, output_dir);
        classify(saliency(:,size(saliency,1)), file_name, data_dir)

    end

end



%%
function saliency = forward(data_dir, file_name, output_dir)
    file_prefix = split(file_name, '.');
    file_prefix = file_prefix{1};

    wav_path = strcat(data_dir, '/', file_name);
    output_path = strcat(output_dir, '/', file_prefix, "_saliency.mat");
    infoaudio = audioinfo(wav_path);
    sdata = audioread(wav_path);
    t = linspace(0, infoaudio.Duration, length(sdata));
    Fs = infoaudio.SampleRate;
    saliency = zeros(int16(infoaudio.Duration), Fs, 1);

    for i = 1:infoaudio.Duration;
        currentClip = reshape(sdata((i-1)*Fs+1: i*Fs, :), Fs, 1);
        W = currentClip(:, 1);
           
        sW = spectrum1DwithMelCepstrumTrial(W, infoaudio,1)';
        windowSize = infoaudio.SampleRate/2;
        b = (1/windowSize)*ones(1,windowSize);
        a = 1;
    
        hW = filter(b, a, sW);
        saliency(i, :, 1) = hW;
    end

    save(output_path, 'saliency', '-v6');
end

function classify(saliency_channel, file_name, data_dir)
    subject_id = extract_subject_id(file_name);
    annotation_file_path = strcat(data_dir, "/", subject_id, ".txt");
    annotation_file = readlines(annotation_file_path);
    
    label = split(annotation_file(11), " ");
    label = label(2);
end

function subject_id = extract_subject_id(file_name)
    pattern = '.' | '_';
    file_name_parts = split(file_name, pattern );
    subject_id = file_name_parts{1};
end