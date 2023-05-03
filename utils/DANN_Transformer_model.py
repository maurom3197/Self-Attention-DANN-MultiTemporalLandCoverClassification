import torch
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer
from torch.nn.modules import LayerNorm, Linear, Sequential, ReLU, GELU

import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

__all__ = ['ViTransformerExtractor']

class ViTransformerExtractor(nn.Module):
    def __init__(self, input_dim=10, num_classes=9, time_dim = 45, d_model=64, n_head=2, n_layers=5,
                 d_inner=128, activation="relu", dropout=0.1):

        super(ViTransformerExtractor, self).__init__()
        self.modelname = f"TransformerEncoder_input-dim={input_dim}_num-classes={num_classes}_" \
                         f"d-model={d_model}_d-inner={d_inner}_n-layers={n_layers}_n-head={n_head}_" \
                         f"dropout={dropout}"

        encoder_layer = TransformerEncoderLayer(d_model, n_head, d_inner, dropout, activation)
        encoder_norm = LayerNorm(d_model)

        self.inlinear = Linear(input_dim, d_model)
        self.relu = ReLU()
        self.transformerencoder = TransformerEncoder(encoder_layer, n_layers, encoder_norm)

        self.pos_embedding = nn.Parameter(torch.randn(1, time_dim, d_model)) # T + class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model)) # class token

        self.dropout = nn.Dropout(dropout)

        self.n_units  = 128

    def forward(self,x):

        x = self.inlinear(x) # B x T x D

        b, n, _ = x.shape # B x T x D

        #cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)  # repeat for all batch
        #x = torch.cat((cls_tokens, x), dim=1) # concatenate on sequence [T + class token]
        x += self.pos_embedding[:, :(n)]
        
        x = x.transpose(0, 1) # N x T x D -> T x N x D
        x = self.transformerencoder(x)
        x = x.transpose(0, 1) # T x N x D -> N x T x D

        features = x.max(1)[0]  # take first dimension B x T x D
        
        return features

# REVERSE GRAD LAYER FOR DANN 
from torch.autograd import Function

class ReverseLayerF(Function):
    # Forwards identity
    # Sends backward reversed gradients
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

# VISION TRANSFORMER DANN MODEL
__all__ = ['ViTransformerDANN']

class ViTransformerDANN(nn.Module):
    def __init__(self, feature_ex, input_dim=13, num_classes=9, d_model=64, n_head=2, n_layers=5, n_domain=2,
                 d_inner=128, activation="relu", dropout=0.1):

        super(ViTransformerDANN, self).__init__()
        self.modelname = f"TransformerEncoder_input-dim={input_dim}_num-classes={num_classes}_" \
                         f"d-model={d_model}_d-inner={d_inner}_n-layers={n_layers}_n-head={n_head}_" \
                         f"dropout={dropout}"

        encoder_layer = TransformerEncoderLayer(d_model, n_head, d_inner, dropout, activation)
        encoder_norm = LayerNorm(d_model)

        self.inlinear = Linear(input_dim, d_model)
        self.relu = ReLU()
        self.transformerencoder = TransformerEncoder(encoder_layer, n_layers, encoder_norm)
        self.n_units  = 128
        self.fc1 = Linear(d_model, self.n_units)
        self.fc2 = Linear(d_model, self.n_units)
        self.outlinear = Linear(self.n_units, num_classes)
        
        self.dropout_p = 0.1
        self.dropout = nn.Dropout(p = self.dropout_p)
        self.outlinear_dom = Linear(self.n_units, n_domain)
        
        self.mlp_head = nn.Sequential(
            LayerNorm(d_model),
            Linear(d_model, self.n_units),
            ReLU(),
            nn.Dropout(p = self.dropout_p),
            Linear(self.n_units, num_classes)
        )

        self.features = feature_ex
    
    def forward(self, x, alpha=None):
        embeddings = self.features(x)

        # If we pass alpha, we can assume we are training the discriminator
        if alpha is not None:
            # gradient reversal layer (backward gradients will be reversed)
            reverse_feature = ReverseLayerF.apply(embeddings, alpha)
            x = self.fc2(reverse_feature)
            x = self.relu(x)
            x = self.dropout(x)
            domain_output = self.outlinear_dom(x)
            return domain_output

        # If we don't pass alpha, we assume we are training with supervision
        else:
            # pass features to labels classifier
            class_logits = self.mlp_head(embeddings)

            return embeddings, class_logits

# STANDARD VISION TRANSFORMER MODEL
__all__ = ['ViTransformer']

class ViTransformer(nn.Module):
    def __init__(self, input_dim=13, num_classes=9, time_dim = 45, d_model=64, n_head=2, n_layers=3,
                 d_inner=128, activation="relu", dropout=0.1):

        super(ViTransformer, self).__init__()
        self.modelname = f"TransformerEncoder_input-dim={input_dim}_num-classes={num_classes}_" \
                         f"d-model={d_model}_d-inner={d_inner}_n-layers={n_layers}_n-head={n_head}_" \
                         f"dropout={dropout}"

        encoder_layer = TransformerEncoderLayer(d_model, n_head, d_inner, dropout, activation)
        encoder_norm = LayerNorm(d_model)

        self.inlinear = Linear(input_dim, d_model)
        self.relu = ReLU()
        self.gelu = GELU()
        self.transformerencoder = TransformerEncoder(encoder_layer, n_layers, encoder_norm)

        self.pos_embedding = nn.Parameter(torch.randn(1, time_dim, d_model)) # T + class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model)) # class token

        self.dropout = nn.Dropout(dropout)

        self.n_units  = 128
        self.fc1 = Linear(d_model, self.n_units)
        
        self.dropout_p = 0.2
        self.dropout2 = nn.Dropout(p = self.dropout_p)
        
        self.mlp_head = nn.Sequential(
            LayerNorm(d_model),
            Linear(d_model, self.n_units),
            ReLU(),
            Linear(self.n_units, num_classes)
        )


    def forward(self,x):

        x = self.inlinear(x) # B x T x D

        b, n, _ = x.shape # B x T x D

        #cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)  # repeat for all batch
        #x = torch.cat((cls_tokens, x), dim=1) # concatenate on sequence [T + class token]
        x += self.pos_embedding[:, :(n)]
        
        x = x.transpose(0, 1) # N x T x D -> T x N x D
        x = self.transformerencoder(x)
        x = x.transpose(0, 1) # T x N x D -> N x T x D
        
        embeddings = x.max(1)[0]
        #x = x[:,0]  # take first dimension B x T x D

        logits = self.mlp_head(embeddings)
        
        logprobs = F.log_softmax(logits, dim=-1)
        
        return embeddings, logits, logprobs 