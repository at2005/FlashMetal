import torch
import torch.nn.functional as F
from math import sqrt
from FlashMetal import FlashAttentionForward, FlashAttentionBackward


n_heads = 4
q = torch.randn(2,n_heads,1024, 96,  requires_grad=True).to("mps")
k = torch.randn(2,n_heads,1024, 96,  requires_grad=True).to("mps")
v = torch.randn(2,n_heads,1024, 96, requires_grad=True).to("mps")

#s = q @ k.transpose(-1,-2)
#s /= sqrt(96)

#mask = torch.tril(torch.ones_like(s)).to("mps")
#s_masked = torch.where(mask == 1, s, torch.tensor(float('-inf')).to("mps"))

#s_masked = F.softmax(s_masked, -1)

#o1 = (s_masked @ v)
#out = ((FlashAttentionMPS(q,k,v)))


class FlashAttentionAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value):
        batch_size, num_heads, N_seq, n_embed = query.size()
        
        out = torch.empty_like(value, requires_grad=True, device='mps')
        row_max = torch.empty((batch_size, num_heads, N_seq), device='mps')
        row_sum = torch.empty((batch_size, num_heads, N_seq), device='mps')

        out, row_max, row_sum =  FlashAttentionForward(query, key, value, out, row_max, row_sum)
        
        ctx.save_for_backward(query, key, value, out, row_max, row_sum)
       	#print(row_max) 
       	#print(row_sum )
        return out

    @staticmethod
    def backward(ctx, grad_output):
        query, key, value, out, row_max, row_sum = ctx.saved_tensors
        
        out_dQ = torch.zeros_like(query, device='mps')
        out_dK = torch.zeros_like(key, device='mps')
        out_dV = torch.zeros_like(value, device='mps')
	        
        res_metal = FlashAttentionBackward(query, key, value, out, grad_output, out_dQ, out_dK, out_dV, row_sum, row_max)
        print(res_metal[0])
        grad_query, grad_key, grad_value = res_metal
        return grad_query, grad_key, grad_value


out = FlashAttentionAutograd.apply(q,k,v)
loss = torch.mean(out)
loss.backward()
#(out.backward(torch.randn_like(out)))
from torch.autograd import gradcheck

test = gradcheck(FlashAttentionAutograd.apply, (q, k, v), eps=1e-6, atol=1e-4)
print("Gradient check passed:", test)

