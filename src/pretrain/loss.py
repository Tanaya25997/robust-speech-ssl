import torch 
import torch.nn as nn 
import torch.nn.functional as F 




# temperature is the τ\tau
# τ from the math — controls how peaked the softmax is. Small τ\tau
# τ (like 0.07) makes the model very confident — it strongly separates positives from negatives. Too small and training becomes unstable. Too large and the loss becomes too soft to learn anything. 0.07 is the standard value from SimCLR

class InfoNCELoss(nn.Module):
    def __init__(self, temperature = 0.07):
        super().__init__()
        self.temperature = temperature 

    
    '''
    p1 and p2 are the projections of the two augmented views of the same clips — both shape [batch, 128]. F.normalize makes every vector unit length (length = 1) so that the dot product between two vectors equals their cosine similarity
    Temperature defines the confidence of the model

    **Small τ\tau
    τ (like 0.07)** — dividing by a small number makes all values larger, so the softmax becomes very peaked. The model is very confident — the positive pair needs to be *way* more similar than the negatives to satisfy the loss. Hard, strict training.
    Large τ\tau
    τ (like 0.5) — values stay small, softmax is flat and soft. The model doesn't need to separate positives and negatives very strongly. Easy, lazy training.
    '''
    def forward(self,p1,p2):
        p1 = F.normalize(p1, dim=1)
        p2 = F.normalize(p2, dim=1)

        # See Note 1 below for this
        batch_size = p1.shape[0]
        p = torch.cat([p1,p2], dim = 0)
        sim = torch.matmul(p, p.T) / self.temperature


        ## diag entries are meaningless since they are dot prod of each vector to itself, hence discard
        mask = torch.eye(2*batch_size, dtype = torch.bool)
        sim = sim.masked_fill(mask, float('-inf'))


        # See Note 2 below 
        labels = torch.cat([
            torch.arange(batch_size) + batch_size,
            torch.arange(batch_size)
        ])

        # See Note 3
        loss = F.cross_entropy(sim, labels)
        return loss 


if __name__=='__main__':
    loss_fn = InfoNCELoss(temperature=0.07)
    p1 = torch.randn(4,128)
    p2 = torch.randn(4,128)
    loss = loss_fn(p1,p2)
    print(f"Loss: {loss.item():.4f}")



'''

Note 1: 

Okay, say batch size N=2. So you have 2 clips, each augmented twice.

```
p1 = [a, b]   # augmentation 1 of clip 1 and clip 2
p2 = [c, d]   # augmentation 2 of clip 1 and clip 2
```

Positive pairs are: (a,c) and (b,d) — same clip, different augmentation.

**If you did `p1 @ p2.T`** you get a 2×2 matrix:

```
     c    d
a [[ac,  ad],
b  [bc,  bd]]
```

You have ac and bd (the positives) but your negatives are only ad and bc — cross pairs between different clips but only across the two augmentation sets. You're missing aa, bb, cc, dd, ab, cd etc. Not enough negatives.

**If you do `p @ p.T`** where `p = [a, b, c, d]` you get a 4×4 matrix:

```
     a    b    c    d
a [[aa,  ab,  ac,  ad],
b  [ba,  bb,  bc,  bd],
c  [ca,  cb,  cc,  cd],
d  [da,  db,  dc,  dd]]
```

Now row `a` has `ac` as the positive and `ab, ad` as negatives. Every clip sees every other clip as a negative. That's the full InfoNCE setup.

The diagonal (aa, bb, cc, dd) we mask out because a clip compared to itself is meaningless.
'''


'''
Note 2:

Okay let's use our concrete example with N=2.

After concatenating, `p = [a, b, c, d]` where indices are:

```
index 0 = a  (clip 1, augmentation 1)
index 1 = b  (clip 2, augmentation 1)
index 2 = c  (clip 1, augmentation 2)
index 3 = d  (clip 2, augmentation 2)
```

Positive pairs are (a,c) and (b,d). So:

```
row 0 (a) → correct answer is index 2 (c)
row 1 (b) → correct answer is index 3 (d)
row 2 (c) → correct answer is index 0 (a)
row 3 (d) → correct answer is index 1 (b)
```

So labels = `[2, 3, 0, 1]`

Now look at the code:

```python
torch.arange(batch_size) + batch_size  # [0,1] + 2 = [2,3]  for rows 0,1
torch.arange(batch_size)               # [0,1]             for rows 2,3
```

Concatenated → `[2, 3, 0, 1]`

It's just saying "for each row, what column is your positive?" Does that click?

'''


'''
Note 3: 


F.cross_entropy takes the similarity matrix [2N, 2N] and the labels [2N] and computes the loss. It's doing exactly the InfoNCE formula we derived mathematically — for each row, maximize the similarity at the correct label position relative to all other positions.
Cross entropy and InfoNCE are mathematically equivalent here — cross entropy is just "given a probability distribution, how much probability did you assign to the correct class." The correct class here is the positive pair.

'''