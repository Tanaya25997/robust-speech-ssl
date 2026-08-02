import os, statistics, sys
import soundfile as sf 
import time 
from multiprocessing import Pool 


def get_info(f):
    info = sf.info(f)
    return info.duration, info.samplerate



output_dir = 'experiments/analysis/parallel'
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, sys.argv[1] if len(sys.argv) > 1 else "data_stats.txt")

data_dir = '/data/LibriSpeech/train-clean-100'

files = []
for root, _, f in os.walk(data_dir):
    for file in f:
        if file.endswith('.flac'):
            files.append(os.path.join(root, file))

print(f"Total files: {len(files)}")


start = time.time() 
num_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4

with Pool(processes=num_workers) as pool:
    results = pool.map(get_info, files)

durations = [r[0] for r in results]
sample_rates = [r[1] for r in results]

end = time.time()
with open(output_file, 'w') as f:
    f.write(f"Total files: {len(files)}\n")
    f.write(f"Workers: {num_workers}\n")
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
print(f"Time taken: {end - start:.2f}s with {num_workers} workers")



'''
heres how it gives a gain with 4 workers 

At any given moment:

CPU 1 → process 1 (file A) or process 3 (file C)
CPU 2 → process 2 (file B) or process 4 (file D)

The OS scheduler rapidly switches between processes on each core. So you get true parallelism for 2 processes at a time (one per core), and the other 2 are ready and waiting.

But here's the key insight for why 4 workers still gives ~3.7x speedup instead of just 2x:

For IO-bound tasks like NFS reads, a process spends most of its time waiting for the disk to respond — not actually using the CPU. So:

Process 1 → sends NFS request → WAITING (CPU free)
CPU → switches to Process 2 → sends NFS request → WAITING (CPU free)  
CPU → switches to Process 3 → sends NFS request → WAITING
CPU → switches to Process 4 → sends NFS request → WAITING
Process 1 → NFS responds → CPU processes result → sends next request

All 4 processes have NFS requests in flight simultaneously even though only 2 CPUs exist. So you're effectively parallelizing the NFS wait time across 4 workers — that's why you get close to 4x speedup despite having only 2 cores.

This is why IO-bound tasks benefit from more workers than cores. CPU-bound tasks (like matrix multiplication) would only benefit up to the number of cores.
'''


'''
note: on compute node with 176 cores thsi script tool 3.74 s on first run but next runs was 0.76 s (cache warm)

(ssl) ubuntu@slurm-login-node-0:~/robust-speech-ssl$ srun -N 1 --ntasks-per-node=1 --cpus-per-task=176 python scripts/explore_data_parallel.py data_stats_compute_p176.txt 176
Total files: 28539
Stats saved to experiments/analysis/parallel/data_stats_compute_p176.txt
Time taken: 3.74s with 176 workers
(ssl) ubuntu@slurm-login-node-0:~/robust-speech-ssl$ 
(ssl) ubuntu@slurm-login-node-0:~/robust-speech-ssl$ 
(ssl) ubuntu@slurm-login-node-0:~/robust-speech-ssl$ srun -N 1 --ntasks-per-node=1 --cpus-per-task=176 python scripts/explore_data_parallel.py data_stats_compute_p176.txt 176
Total files: 28539
Stats saved to experiments/analysis/parallel/data_stats_compute_p176.txt
Time taken: 0.76s with 176 workers
(ssl) ubuntu@slurm-login-node-0:~/robust-speech-ssl$ 
(ssl) ubuntu@slurm-login-node-0:~/robust-speech-ssl$ 
(ssl) ubuntu@slurm-login-node-0:~/robust-speech-ssl$ srun -N 1 --ntasks-per-node=1 --cpus-per-task=176 python scripts/explore_data_parallel.py data_stats_compute_p176.txt 176
Total files: 28539
Stats saved to experiments/analysis/parallel/data_stats_compute_p176.txt
Time taken: 4.36s with 176 workers
(ssl) ubuntu@slurm-login-node-0:~/robust-speech-ssl$ srun -N 1 --ntasks-per-node=1 --cpus-per-task=176 python scripts/explore_data_parallel.py data_stats_compute_p176.txt 176
Total files: 28539
Stats saved to experiments/analysis/parallel/data_stats_compute_p176.txt
Time taken: 3.45s with 176 workers
(ssl) ubuntu@slurm-login-node-0:~/robust-speech-ssl$ srun -N 1 --ntasks-per-node=1 --cpus-per-task=176 python scripts/explore_data_parallel.py data_stats_compute_p176.txt 176
Total files: 28539
Stats saved to experiments/analysis/parallel/data_stats_compute_p176.txt
Time taken: 0.75s with 176 workers

cold -> hot -> cold -> cold -> hot (different compute node assigend)


ssl) ubuntu@slurm-login-node-0:~/robust-speech-ssl$ srun -N 1 --ntasks-per-node=1 --cpus-per-task=176 -w slurm-compute-node-0 python scripts/explore_data_parallel.py data_stats_compute_p176.txt 176
Total files: 28539
Stats saved to experiments/analysis/parallel/data_stats_compute_p176.txt
Time taken: 3.69s with 176 workers
(ssl) ubuntu@slurm-login-node-0:~/robust-speech-ssl$ srun -N 1 --ntasks-per-node=1 --cpus-per-task=176 -w slurm-compute-node-0 python scripts/explore_data_parallel.py data_stats_compute_p176.txt 176
Total files: 28539
Stats saved to experiments/analysis/parallel/data_stats_compute_p176.txt
Time taken: 0.76s with 176 workers
(ssl) ubuntu@slurm-login-node-0:~/robust-speech-ssl$ srun -N 1 --ntasks-per-node=1 --cpus-per-task=176 -w slurm-compute-node-0 python scripts/explore_data_parallel.py data_stats_compute_p176.txt 176
Total files: 28539
Stats saved to experiments/analysis/parallel/data_stats_compute_p176.txt
Time taken: 0.76s with 176 workers
(ssl) ubuntu@slurm-login-node-0:~/robust-speech-ssl$ srun -N 1 --ntasks-per-node=1 --cpus-per-task=176 -w slurm-compute-node-0 python scripts/explore_data_parallel.py data_stats_compute_p176.txt 176
Total files: 28539
Stats saved to experiments/analysis/parallel/data_stats_compute_p176.txt
Time taken: 0.75s with 176 workers
(ssl) ubuntu@slurm-login-node-0:~/robust-speech-ssl$ srun -N 1 --ntasks-per-node=1 --cpus-per-task=176 -w slurm-compute-node-0 python scripts/explore_data_parallel.py data_stats_compute_p176.txt 176
Total files: 28539
Stats saved to experiments/analysis/parallel/data_stats_compute_p176.txt
Time taken: 3.45s with 176 workers
(ssl) ubuntu@slurm-login-node-0:~/robust-speech-ssl$ 

last command tried after 2 minutes. cache eviction is pretty quick!

(ssl) ubuntu@slurm-login-node-0:~/robust-speech-ssl$ srun -N 1 --ntasks-per-node=1 --cpus-per-task=176 -w slurm-compute-node-0 python scripts/explore_data_parallel.py data_stats_compute_p100.txt 100
Total files: 28539
Stats saved to experiments/analysis/parallel/data_stats_compute_p100.txt
Time taken: 0.69s with 100 workers
(ssl) ubuntu@slurm-login-node-0:~/robust-speech-ssl$ srun -N 1 --ntasks-per-node=1 --cpus-per-task=176 -w slurm-compute-node-0 python scripts/explore_data_parallel.py data_stats_compute_p100.txt 100
Total files: 28539
Stats saved to experiments/analysis/parallel/data_stats_compute_p100.txt
Time taken: 0.72s with 100 workers



(ssl) ubuntu@slurm-login-node-0:~/robust-speech-ssl$ srun -N 1 --ntasks-per-node=1 --cpus-per-task=176 -w slurm-compute-node-0 python scripts/explore_data_parallel.py data_stats_compute_p352.txt 352
Total files: 28539
Stats saved to experiments/analysis/parallel/data_stats_compute_p352.txt
Time taken: 3.57s with 352 workers
(ssl) ubuntu@slurm-login-node-0:~/robust-speech-ssl$ srun -N 1 --ntasks-per-node=1 --cpus-per-task=176 -w slurm-compute-node-0 python scripts/explore_data_parallel.py data_stats_compute_p352.txt 352
Total files: 28539
Stats saved to experiments/analysis/parallel/data_stats_compute_p352.txt
Time taken: 0.97s with 352 workers
(ssl) ubuntu@slurm-login-node-0:~/robust-speech-ssl$ srun -N 1 --ntasks-per-node=1 --cpus-per-task=176 -w slurm-compute-node-0 python scripts/explore_data_parallel.py data_stats_compute_p352.txt 352
Total files: 28539
Stats saved to experiments/analysis/parallel/data_stats_compute_p352.txt
Time taken: 0.98s with 352 workers

(ssl) ubuntu@slurm-login-node-0:~/robust-speech-ssl$ srun -N 1 --ntasks-per-node=1 --cpus-per-task=176 -w slurm-compute-node-0 python scripts/explore_data_parallel.py data_stats_compute_p1056.txt 1056
Total files: 28539
Stats saved to experiments/analysis/parallel/data_stats_compute_p1056.txt
Time taken: 5.34s with 1056 workers
(ssl) ubuntu@slurm-login-node-0:~/robust-speech-ssl$ srun -N 1 --ntasks-per-node=1 --cpus-per-task=176 -w slurm-compute-node-0 python scripts/explore_data_parallel.py data_stats_compute_p1056.txt 1056
Total files: 28539
Stats saved to experiments/analysis/parallel/data_stats_compute_p1056.txt
Time taken: 2.37s with 1056 workers
(ssl) ubuntu@slurm-login-node-0:~/robust-speech-ssl$ srun -N 1 --ntasks-per-node=1 --cpus-per-task=176 -w slurm-compute-node-0 python scripts/explore_data_parallel.py data_stats_compute_p1056.txt 1056
Total files: 28539
Stats saved to experiments/analysis/parallel/data_stats_compute_p1056.txt
Time taken: 2.39s with 1056 workers
(ssl) ubuntu@slurm-login-node-0:~/robust-speech-ssl$ 


noet: context switching with workers > cpu cores is only adding iverhead here. possible because nfs bandwidth is alreday saturated 
'''