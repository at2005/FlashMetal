import torch
import torch.nn.functional as F
from torch import nn
from math import sqrt
from FlashMetal import FlashAttentionForward, FlashAttentionBackward, fetchPipeline

forward_pip, backward_pip = fetchPipeline()


n_embed = 384
n_heads = 4


q = torch.randn(2,n_heads,1024, 96,  requires_grad=True, device="mps")
k = torch.randn(2,n_heads,1024, 96,  requires_grad=True, device="mps")
v = torch.randn(2,n_heads,1024, 96, requires_grad=True, device="mps")


s = q @ k.transpose(-1,-2)
s /= sqrt(96)

mask = torch.tril(torch.ones_like(s)).to("mps")
s_masked = torch.where(mask == 1, s, torch.tensor(float('-inf')).to("mps"))

P = F.softmax(s_masked, -1)

o_test = (torch.matmul(P, v))


#dO = torch.randn_like(q, device='mps')
#o1 = (s_masked @ v)
#dP = torch.matmul(dO, v.transpose(-1, -2))
#dS = P * (dP - torch.sum(dP * P, dim=-1, keepdim=True))

#print(torch.matmul(dS, k))



class FlashAttentionAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value):
        batch_size, num_heads, N_seq, n_embed = query.size()
        
        out = torch.empty_like(value, requires_grad=True, device='mps')
        row_max = torch.empty((batch_size, num_heads, N_seq), device='mps')
        row_sum = torch.empty((batch_size, num_heads, N_seq), device='mps')

        out, row_max, row_sum =  FlashAttentionForward(query, key, value, out, row_max, row_sum, forward_pip)
        
        ctx.save_for_backward(query, key, value, out, row_max, row_sum)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        query, key, value, out, row_max, row_sum = ctx.saved_tensors
        
        out_dQ = torch.zeros_like(query, device='mps')
        out_dK = torch.zeros_like(key, device='mps')
        out_dV = torch.zeros_like(value, device='mps')
        res_metal = FlashAttentionBackward(query, key, value, out, grad_output, out_dQ, out_dK, out_dV, row_sum, row_max, backward_pip)
        grad_query, grad_key, grad_value = res_metal
        return grad_query, grad_key, grad_value


out = FlashAttentionAutograd.apply(q,k,v)
#print(out)
#diff = out - o_test

#print(diff)

loss = torch.mean(out)
loss.backward()
print(q.grad)

class MHAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.head_size = 96
        self.batch_qkv_matrices = nn.Linear(n_embed, self.head_size * n_heads * 3, bias=False)
        self.projection = nn.Linear(n_embed, n_embed)
        # self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B,T,C = x.shape
        q,k,v = self.batch_qkv_matrices(x).split(self.head_size * n_heads, dim=-1) # Now Q,K,V of dim B, T, head size * n_heads
        q = q.view(B, T, n_heads, self.head_size).transpose(1,2) # Now of shape B, n_heads, T, head_size for BMM
        k = k.view(B, T, n_heads, self.head_size).transpose(1,2)
        v = v.view(B, T, n_heads, self.head_size).transpose(1,2)
        
        out = FlashAttentionAutograd.apply(q,k,v)
        

        return out

#x = torch.randn(1,1024, 384, device="mps")
#mh = MHAttention().to("mps")
#out = mh(x)
#loss = torch.mean(out)
#loss.backward()


#print("Gradient check passed:", test)
