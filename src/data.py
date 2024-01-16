import torch
from torchaudio import datasets
import os
import webrtcvad
import pandas
import numpy as np
from src.audio_signal import AudioSignal

class LibriSpeech:
    """Handles LibriSpeech data including audio waveforms, sample rate, spoken text, and index info."""

    FRAME_SIZE_MS = 10  

    def __init__(self, root_dir, dataset_name, sub_dir, download=False):
        """Initializes LibriSpeech dataset.

        Args:
            root_dir (str): Path to the project's root directory.
            dataset_name (str): Dataset name (e.g., "dev-clean", "dev-other").
            sub_dir (str): Subdirectory within archive ('LibriSpeech').
            download (bool): If True, downloads the dataset if not found.
        """
        self.dataset = datasets.LIBRISPEECH(
            root=root_dir, url=dataset_name, folder_in_archive=sub_dir, download=download
        )
        self.name = dataset_name
        dataset_path = self.dataset._path
        if not os.path.exists(dataset_path):
            raise RuntimeError(
                "Dataset not found. Please set 'download=True' or manually download from https://www.openslr.org/12."
            )

    def load_data(self, index, n_mels, n_mfcc):
        """Loads data for a given index from the dataset.

        Args:
            index (int): Index of data in the dataset.
            n_mels (int): Number of mel bins for MFCC extraction.
            n_mfcc (int): Number of MFCC coefficients.

        Returns:
            tuple: MFCC data (tensor) and label data (tensor).
        """
        # Extracting MFCC features
        audio_data, sample_rate = self.dataset[index][:2]
        signal = Signal(audio_data, sample_rate)
        frame_size = int(sample_rate * (self.FRAME_SIZE_MS / 1000.0))
        X = signal.get_MFCC(hop_length=frame_size, n_mels=n_mels, n_mfcc=n_mfcc).transpose(2,0).transpose(1,2)

        # Extracting labels
        label_path = f"{os.getcwd()}/LibriSpeech/labels/{self.name}/{self.dataset[index][3]}/{self.dataset[index][4]}"
        label_file = f"{label_path}/{self.dataset[index][3]}-{self.dataset[index][4]}-{str(self.dataset[index][5]).zfill(4)}.csv"
        y = torch.tensor(pandas.read_csv(label_file, delimiter=",", header=None).values)

        return X, y

    def _label_data(self, dataset_name, verbose=False):
        """Generates labels for dataset and saves them as CSV files.

        Args:
            dataset_name (str): Dataset name (e.g., "dev-clean").
            verbose (bool): If True, displays progress in CLI.
        """
        vad = webrtcvad.Vad(1)
        label_dir = f"{os.getcwd()}/LibriSpeech/labels/{dataset_name}"

        for i, data in enumerate(self.dataset):
            signal = Signal(data[0], data[1])
            frame_size = int(signal.sample_rate * (self.FRAME_SIZE_MS / 1000.0))
            split_waveform = signal.split_into_frames(frame_size=frame_size)
            labels = [1 if vad.is_speech(np.int16(f * 32768).tobytes(), sample_rate=signal.sample_rate) else 0 for f in split_waveform]

            file_label_dir = f"{label_dir}/{self.dataset[i][3]}/{self.dataset[i][4]}"
            os.makedirs(file_label_dir, exist_ok=True)

            file_name = f"{file_label_dir}/{self.dataset[i][3]}-{self.dataset[i][4]}-{str(self.dataset[i][5]).zfill(4)}.csv"
            if verbose:
                print(f"Writing labels to {file_name}...")
            
            with open(file_name, "w") as label_file:
                label_file.write(','.join(map(str, labels)))

def build_librispeech(mode, verbose=False):
    """Creates LibriSpeech objects for each dataset.

    Args:
        mode (str): 'training' or 'testing'.
        verbose (bool): If True, shows progress in CLI.

    Returns:
        dict: Dictionary of LibriSpeech objects.
    """
    if mode not in ['training', 'testing']:
        raise ValueError("Mode must be 'training' or 'testing'.")

    librispeech = {}
    datasets = ["dev-clean", "dev-other"]
