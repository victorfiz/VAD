import torch
import librosa
import torchaudio.transforms as T
import matplotlib.pyplot as plt

class AudioSignal:
    """
    Represents an audio signal and provides methods for signal processing.
    """

    def __init__(self, waveform, sample_rate):
        self.waveform = waveform
        self.sample_rate = sample_rate

    def split_frames(self, frame_size=1024):
        """
        Splits the audio signal into frames of the specified size.
        """
        num_frames = self.waveform.size(1) // frame_size
        return self.waveform.unfold(1, frame_size, frame_size).narrow(1, 0, num_frames)

    def extract_mfcc(self, hop_length, n_mels=256, n_mfcc=256):
        """
        Extracts MFCC features from the audio signal.
        """
        mfcc_transform = T.MFCC(sample_rate=self.sample_rate, n_mfcc=n_mfcc,
                                melkwargs={'n_fft': 2048, 'n_mels': n_mels, 'hop_length': hop_length})
        return mfcc_transform(self.waveform)

    def extract_spectrogram(self, n_fft=1024, hop_length=512):
        """
        Extracts a spectrogram from the audio signal.
        """
        spectrogram = T.Spectrogram(n_fft=n_fft, win_length=n_fft, hop_length=hop_length, power=2)
        return spectrogram(self.waveform)

    def extract_pitch(self):
        """
        Extracts pitch features from the audio signal.
        """
        pitch_feature = torchaudio.functional.compute_kaldi_pitch(self.waveform, self.sample_rate)
        pitch, nfcc = pitch_feature[..., 0], pitch_feature[..., 1]
        return pitch, nfcc

# Visualization Functions
def plot_spectrogram(spec, title='Spectrogram'):
    """
    Plots a spectrogram.
    """
    plt.figure(figsize=(10, 4))
    plt.imshow(librosa.power_to_db(spec), aspect='auto', origin='lower')
    plt.title(title)
    plt.ylabel('Frequency bins')
    plt.xlabel('Time frames')
    plt.colorbar(format='%+2.0f dB')
    plt.show()

def plot_mel_filterbank(fbank, title='Mel Filter Bank'):
    """
    Plots a mel filter bank.
    """
    plt.figure(figsize=(10, 4))
    plt.imshow(fbank.detach().numpy(), aspect='auto', origin='lower')
    plt.title(title)
    plt.ylabel('Mel filters')
    plt.xlabel('Frequency bins')
    plt.colorbar()
    plt.show()

def plot_kaldi_pitch(waveform, sample_rate, pitch, nfcc):
    """
    Plots Kaldi pitch features.
    """
    plt.figure(figsize=(10, 4))
    time_axis = torch.linspace(0, waveform.shape[1] / sample_rate, waveform.shape[1])
    plt.plot(time_axis, waveform[0].numpy(), linewidth=1, color='gray', alpha=0.3)

    pitch_time_axis = torch.linspace(0, time_axis[-1], pitch.shape[1])
    plt.plot(pitch_time_axis, pitch[0].numpy(), linewidth=2, label='Pitch', color='green')

    nfcc_time_axis = torch.linspace(0, time_axis[-1], nfcc.shape[1])
    plt.plot(nfcc_time_axis, nfcc[0].numpy(), linewidth=2, label='NFCC', color='blue', linestyle='--')

    plt.title("Kaldi Pitch Feature")
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()
