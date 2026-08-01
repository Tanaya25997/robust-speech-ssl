import torchaudio 
import os 
import statistics
import soundfile as sf

output_dir = 'experiments/analysis'
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "data_stats.txt")


data_dir = '/data/LibriSpeech/train-clean-100'

files = []

for root, _, f in os.walk(data_dir):
    for file in f: 
        if file.endswith('.flac'):
            files.append(os.path.join(root, file))


print(f"Total files: {len(files)}")

durations = []
sample_rates = []

for f in files[:500]:
    info = sf.info(f)
    #duration =   #info.num_frames / info.sample_rate ### samples = time * samplling_frequency
    durations.append(info.duration)
    sample_rates.append(info.samplerate)

## descriptive stats 


with open(output_file, 'w') as f:
    f.write(f"Total files: {len(files)} \n")
    f.write(f"Sample rates found: {set(sample_rates)}\n")
    f.write(f"Min duration:  {min(durations):.2f}s\n")
    f.write(f"Max duration:  {max(durations):.2f}s\n")
    f.write(f"Mean duration: {sum(durations)/len(durations):.2f}s\n")
    f.write(f"Median duration: {statistics.median(durations):.2f}s\n")
    f.write(f"Std duration:    {statistics.stdev(durations):.2f}s\n")
    f.write(f"Files under 3s: {sum(1 for d in durations if d < 3)}\n")
    f.write(f"Files over 3s:  {sum(1 for d in durations if d >= 3)}\n")


print(f"Stats saved to {output_file}")