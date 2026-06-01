import torch 
import torch.nn as nn


class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
 

        ## (out_channels, kernel_size, stride)
        conv_layers = [
            (512, 10, 5),
            (512, 3, 2),
            (512, 3, 2),
            (512,  3, 2),
            (512,  3, 2),
            (512,  2, 2),
            (512,  2, 2),
        ]

        layers = []
        in_channels = 1 ## mono audio 

        for out_channels, kernel, stride in conv_layers:
            layers.append(
                nn.Conv1d(in_channels,out_channels,kernel_size=kernel,stride=stride)
            )
            in_channels = out_channels
            layers.append(nn.LayerNorm(out_channels))
            layers.append(nn.GELU())

        self.conv = nn.Sequential(*layers)
        self.out_channels = 512


    def forward(self,x):
        if x.dim() == 2: # [batch, time]
            x = x.unsqueeze(1) # [batch, 1, time]

        for layer in self.conv:
            if isinstance(layer, nn.LayerNorm):
                x = x.transpose(1,2)
                x = layer(x)
                x = x.transpose(1,2)
            else:
                x = layer(x)
        return x
    


'''
The CNN gave us [512, 49] for one clip — 49 timesteps, each with 512 features. But the CNN processed each timestep independently. It has no idea how timestep 5 relates to timestep 30.
That's the Transformer's job — look at all 49 timesteps together and let them talk to each other. This is the attention mechanism. Timestep 5 can say "hey I'm related to timestep 30" and update its representation accordingly.
So the Transformer input and output are both [49, 512] — same shape in, same shape out. It doesn't compress anything, it just enriches each timestep's representation using context from all other timesteps.
'''

class TransformerEncoder(nn.Module):
    def __init__(self, d_model = 512, nhead=8, num_layers=6): ##d_model == cnn output channels, nhead = the number of attention heads, num_layers = 6 stacked transformer blocks
        super().__init__()


        '''
        Note: nn.TransformerEncoderLayer is PyTorch's built-in single transformer block — attention + feedforward + layernorm all in one. dim_feedforward=2048 is the size of the inner feedforward layer (usually 4× d_model). batch_first=True means our tensors are [batch, time, features]
        '''
        encoder_layer = nn.TransformerEncoderLayer(
            d_model = d_model,
            nhead = nhead,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True    ### [batch, timesteps/time, features]
        )
    
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.norm = nn.LayerNorm(d_model)
        
    def forward(self,x):
        # x is in [batch, channels, time] = [batch, 512, 49]  -> output of CNN
        x = x.transpose(1,2) # [batch, time, channels] =  [batch, 49, 512] 
        x = self.transformer(x)
        x = self.norm(x)
        ## output is same as input -> [batch, time, channels] but with richer representations
        return x
    

if __name__ == '__main__':
    
    ## test CNN loop 
    cnn = CNNFeatureExtractor()
    transformer = TransformerEncoder()

    dummy = torch.randn(4, 16000)
    print(f"Input: {dummy.shape}") ## Input: torch.Size([4, 16000])

    cnn_out = cnn(dummy)
    print(f"Output: {cnn_out.shape}")  ## Output: torch.Size([4, 512, 49]) --> this is 49 timsteps each of 512 values. so basically 16000 gets downsampled to 49 and now instead of just 1 value per timestep (the amplitude) we now have 512 features per timestep!

    trans_out = transformer(cnn_out)
    print(f"Transformer output: {trans_out.shape}")