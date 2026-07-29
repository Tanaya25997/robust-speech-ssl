import torch 
from torch.utils.data import Dataset 
import torchaudio 
import os 

class SpeechDataset(Dataset):

    def __init__(self, root_dir, sample_rate=16000, duration=3):
        self.root_dir = root_dir 
        self.sample_rate = sample_rate   ### 16khx since speech is mostly in 80Hz-8kHz, Fun fact: music is samples at 44.1kHz
        self.duration = duration 
        self.samples = duration * sample_rate 

        self.files = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.flac') or file.endswith('.wav'):
                    self.files.append(os.path.join(root,file))

    def __len__(self):
        return len(self.files) # len(dataset) call this madatory for pytorch Dataset 


    def __getitem__(self, idx):
        path = self.files[idx]
        waveform, sr = torchaudio.load(path)

        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        ## convert to mono 
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        waveform = waveform.squeeze(0) ## change shape from [1, samples] tp [samples]
        if waveform.shape[0] >= self.samples:
            waveform = waveform[:self.samples]
        else:
            pad = self.samples - waveform.shape[0]
            waveform = torch.nn.functional.pad(waveform, (0,pad))

        return waveform 

if __name__=='__main__':
    dataset = SpeechDataset(root_dir="data/raw")
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample shape: {sample.shape}")
    

    


