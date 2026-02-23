from torch import nn, Tensor
from torch.nn import Module, GRU
from torch_einops_utils.save_load import save_load
from einops import repeat

# helper

def exists(v):
    return v is not None

@save_load()
class CompactSequenceEmbedder(Module):
    def __init__(
        self,
        dim,
        **kwargs
    ):
        super().__init__()
        self.gru = GRU(dim, dim, batch_first = True, **kwargs)

    def forward(
        self,
        x,
        mask = None,
        episode_lens: Tensor | None = None
    ):
        seq_len = x.shape[1]

        if exists(episode_lens):
            lengths_cpu = episode_lens.cpu().long()
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths_cpu, batch_first = True, enforce_sorted = False
            )

        _, h = self.gru(x)
        
        # take last hidden state (B, D)
        last_h = h[-1]
        
        # repeat to match sequence length (B, L, D)
        return repeat(last_h, 'b d -> b n d', n = seq_len)
