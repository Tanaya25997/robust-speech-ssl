import torchaudio 
import os,sys
import statistics
import soundfile as sf
import time

output_dir = 'experiments/analysis'
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, sys.argv[1] if len(sys.argv) > 1 else "data_stats.txt")


data_dir = '/data/LibriSpeech/train-clean-100'

files = []

for root, _, f in os.walk(data_dir):
    for file in f: 
        if file.endswith('.flac'):
            files.append(os.path.join(root, file))


print(f"Total files: {len(files)}")

durations = []
sample_rates = []

start = time.time()
for i, f in enumerate(files):
    if i % 1000 == 0:
        print(f"Processing {i}/{len(files)}...")
    info = sf.info(f)
    #duration =   #info.num_frames / info.sample_rate ### samples = time * samplling_frequency
    durations.append(info.duration)
    sample_rates.append(info.samplerate)


end = time.time()

## descriptive stats 


with open(output_file, 'w') as f:
    f.write(f"Total files: {len(files)} \n")
    f.write(f"Time taken: {end - start:.2f}s\n")
    f.write(f"Sample rates found: {set(sample_rates)}\n")
    f.write(f"Min duration:  {min(durations):.2f}s\n")
    f.write(f"Max duration:  {max(durations):.2f}s\n")
    f.write(f"Mean duration: {sum(durations)/len(durations):.2f}s\n")
    f.write(f"Median duration: {statistics.median(durations):.2f}s\n")
    f.write(f"Std duration:    {statistics.stdev(durations):.2f}s\n")
    f.write(f"Files under 3s: {sum(1 for d in durations if d < 3)}\n")
    f.write(f"Files over 3s:  {sum(1 for d in durations if d >= 3)}\n")


print(f"Stats saved to {output_file}")
print(f"Time taken: {end - start:.2f}s")