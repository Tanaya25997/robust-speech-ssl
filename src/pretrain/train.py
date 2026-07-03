'''
model is as follows

  f_theta     g_phi
x----------> z--------->p -----> L


x = raw audio
z = representation, output of encoder [batch, 512]
p = projection, output of projection head [batch, 128]
L = the InfoNCE loss, a single scalar number
'''




import torch 
import torch.nn as nn 
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler 
import sys,os 
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.ssl_model import SSLModel 
from pretrain.loss import InfoNCELoss 


def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    dist.init_process_group(
        backend='nccl',
        rank=rank,
        world_size=world_size
    )


def cleanup_ddp():
    dist.destroy_process_group() 


def augment(x, noise_factor=0.005):
    noise = torch.randn_like(x)*noise_factor 
    return x + noise 

def validate(model, rank, batch_size=32):
    model.eval()
    with torch.no_grad():
        val_data = torch.randn(64, 16000).cuda(rank)
        z, _ = model(val_data)
        std = z.std(dim=0).mean().item()
        if rank == 0:
            print(f"Validation - Representation std: {std:.4f}")

        val_loss_fn = InfoNCELoss(temperature = 0.07)
        x1_val = augment(val_data)
        x2_val = augment(val_data)
        _, p1_val = model(x1_val)
        _, p2_val = model(x2_val)
        val_loss = val_loss_fn(p1_val, p2_val)
        if rank == 0:
            print(f"Validation Loss : {val_loss.item():.4f} | Std: {std:.4f}")

    

def train(rank, world_size, epochs, batch_size=32):
    setup_ddp(rank, world_size)
    torch.cuda.set_device(rank)

    model = SSLModel().cuda(rank)
    model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    loss_fn = InfoNCELoss(temperature=0.07)

    torch.manual_seed(42)
    dataset = torch.randn(1000, 16000)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    for epoch in range(epochs):
        model.train()
        for batch in dataloader:
            x = batch.cuda(rank, non_blocking=True) #blocking could be better in some cases as it will tell the cpu to wait before it can read the value that was sent to gpu. for large sensors, if you try to read asap you could get garbage 
            
            # x1[i] and x2[i] are two views of the same clip 
            x1 = augment(x) # [32,160000] 
            x2 = augment(x) # [32,160000]

            # we only need p during training, hence dicard z 
            _, p1 = model(x1) # [32,128]
            _, p2 = model(x2) # [32,128]

            loss = loss_fn(p1,p2)
            optimizer.zero_grad()
            loss.backward() ### compute gradients and sync across all gpus
            optimizer.step()

            if rank == 0:
                print(f"Epoch {epoch} Loss: {loss.item():.4f}")

        if epoch % 10 == 0:
            validate(model, rank)
        

    cleanup_ddp()


def main():
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size, 100), nprocs=world_size)

if __name__ == '__main__':
    main()

    


