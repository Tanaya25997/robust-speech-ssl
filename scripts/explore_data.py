import torchaudio 
import os 
import statistics


data_dir = '/data/LibriSpeech/train-clean-500'

files = []

for root, _, f in os.walk(data_dir):
    for file in f: 
        if file.endswith('.flac'):
            files.append(os.path.join(root, file))


print(f"Total files: {len(files)}")

durations = []
sample_rates = []

for f in files[:500]:
    info = torchaudio.info(f)
    duration = info.num_frames / info.sample_rate ### samples = time * samplling_frequency
    durations.append(duration)
    sample_rates.append(info.sample_rate)

## descriptive stats 


print(f"Sample rates found: {set(sample_rates)}")
print(f"Min duration:  {min(durations):.2f}s")
print(f"Max duration:  {max(durations):.2f}s")
print(f"Mean duration: {sum(durations)/len(durations):.2f}s")
print(f"Files under 3s: {sum(1 for d in durations if d < 3)}")
print(f"Files over 3s:  {sum(1 for d in durations if d >= 3)}")
print(f"Median duration: {statistics.median(durations):.2f}s")
print(f"Std duration:    {statistics.stdev(durations):.2f}s")