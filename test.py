import torch
import torch.nn.functional as F
from math import sqrt
from FlashMetal import FlashAttentionMPS
q = torch.randn(1,1,1024, 96).to("mps")
k = torch.randn(1,1,1024, 96).to("mps")
v = torch.randn(1,1,1024, 96).to("mps")

s = q @ k.transpose(-1,-2)
s /= sqrt(96)

mask = torch.tril(torch.ones_like(s)).to("mps")
s_masked = torch.where(mask == 1, s, torch.tensor(float('-inf')).to("mps"))

s_masked = F.softmax(s_masked, -1)

o1 = (s_masked @ v)

o2 = (FlashAttentionMPS(q,k,v))
print(o2 - o1)
