import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from encoder import CNNFeatureExtractor, TransformerEncoder


##projection_dim=128 is the size of the space where the contrastive loss will be computed — remember gϕg_\phi
## gϕ​ maps from 512 down to 128

class SSLModel(nn.Module):
    def __init__(self, d_model=512, projection_dim=128):     
        super().__init__()
        self.cnn = CNNFeatureExtractor()
        self.transformer = TransformerEncoder(d_model=d_model)


        self.projection = nn.Sequential(
            nn.Linear(d_model,d_model),
            nn.ReLU(),
            nn.Linear(d_model, projection_dim)
        )

    '''
    We return both z and p — z is the representation you keep for downstream tasks, p is what the contrastive loss operates on. Notice we return both because during pretraining you need p for the loss, but during evaluation you only use z
    '''
    def forward(self,x):
        x = self.cnn(x)  #[batch, 16000] 
        x = self.transformer(x) # [batch, 49, 512]
        z = x.mean(dim=1) # [batch, 512]. representation vector z
        p = self.projection(z)
        return z,p 


if __name__=='__main__':
    model = SSLModel()
    dummy = torch.randn(4,16000)
    z,p = model(dummy)
    print(f"Input: {dummy.shape}")
    print(f"z (representation): {z.shape}")
    print(f"p (projection): {p.shape}")