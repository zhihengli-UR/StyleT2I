import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence


class Sentence2DeltaLatent(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        g_latent_dim,
        hidden_dim=None,
        num_mlp_layers=3,
        return_delta=False,
    ):
        super(Sentence2DeltaLatent, self).__init__()
        assert num_mlp_layers >= 2, "num_mlp_layers must be greater than 2"

        if g_latent_dim is None:
            g_latent_dim = out_dim
        self.g_latent_dim = g_latent_dim

        if hidden_dim is None:
            hidden_dim = out_dim

        self.hidden_dim = hidden_dim
        self.num_layers = 1  # must be one, otherwise it cannot learn anything
        self.num_directions = 2
        self.recurrent = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=self.num_layers,
            bidirectional=self.num_directions == 2,
            batch_first=True,
        )
        fc_in_dim = hidden_dim + g_latent_dim
        layers = [nn.Linear(fc_in_dim, out_dim), nn.ReLU(inplace=True)]

        # add hidden layers
        for _ in range(num_mlp_layers - 2):
            layers.append(nn.Linear(out_dim, out_dim))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Linear(out_dim, out_dim))
        self.mlp = nn.Sequential(*layers)

        self.return_delta = return_delta

    def forward(self, rand_latent_code, text_embed, text_len=None, enforce_sorted=True):
        bs = text_embed.shape[0]
        if text_len is not None:
            text_len = text_len.cpu()
            text_embed = pack_padded_sequence(
                text_embed,
                text_len,
                batch_first=True,
                enforce_sorted=enforce_sorted,
            )
        out = self.recurrent(text_embed)[1]
        out = out.reshape(self.num_layers, self.num_directions, bs, self.hidden_dim)
        out = out.permute(2, 0, 1, 3)  # B x Lx D x C
        out = out[:, -1]  # B x D x C
        sentence_feat = out.mean(dim=1)  # B x C

        concat_feat = torch.cat([sentence_feat, rand_latent_code], dim=1)
        out = self.mlp(concat_feat)
        delta = out

        if (delta.shape[-1] // self.g_latent_dim) == 1:
            # w space
            final_latent_code = rand_latent_code + delta
        else:
            # wp space
            batch_size = rand_latent_code.shape[0]
            final_latent_code = rand_latent_code.unsqueeze(1) + delta.reshape(
                batch_size, -1, self.g_latent_dim
            )

        if not enforce_sorted:
            assert text_len is not None
            final_latent_code = final_latent_code[text_embed.unsorted_indices]
            delta = delta[text_embed.unsorted_indices]

        if self.return_delta:
            return final_latent_code, delta
        else:
            return final_latent_code
