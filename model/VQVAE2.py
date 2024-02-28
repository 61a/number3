"""
date:
author:
"""
import warnings
warnings.filterwarnings("ignore")
import torch
# from model.BetaVAE import BaseVAE
from model.VQVAE.beta_vae import BaseVAE
from torch import nn
from torch.nn import functional as F
from model.VQVAE.types_ import *
from torch.distributions import Categorical
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
class VectorQuantizer(nn.Module):
    """
    Reference:
    [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 beta: float = 0.25):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, latents: Tensor) -> Tensor:
        # latents = latents.permute(0, 2, 3, 1).contiguous()  # [B x D x H x W] -> [B x H x W x C]
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)  # [BHW x D]

        # Compute L2 distance between latents and embedding weights
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim=0, keepdim=True) - \
               2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BHW x K]

        # Get the encoding that has the min distance
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BHW, 1]
        indices = encoding_inds.view(*latents.shape[:3]).unsqueeze(dim=1)  # BHW to BCHW
        # Convert to one-hot encodings
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BHW x K]

        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BHW, D]
        quantized_latents = quantized_latents.view(latents_shape)  # [B x H x W x D]

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        vq_loss = commitment_loss * self.beta + embedding_loss

        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()
        # print('quantized_latents:',quantized_latents.shape)

        return quantized_latents, vq_loss,encoding_one_hot,indices  # [B x C x H x W]
    def embed(self, indices):
        quantized = self.embedding(indices)
        return quantized


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x




class SEBlock(nn.Module):
    def __init__(self, channels,mode='max',  ratio=2):
        super(SEBlock, self).__init__()
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_pooling = nn.AdaptiveMaxPool2d(1)
        if mode == "max":
            self.global_pooling = self.max_pooling
        elif mode == "avg":
            self.global_pooling = self.avg_pooling
        self.fc_layers = nn.Sequential(
            nn.Conv2d(channels, channels * ratio, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(channels * ratio, channels, kernel_size=1),
        )
        # self.sigmoid = nn.Sigmoid()
     
    
    def forward(self, x):
        b, c, _, _ = x.shape
        v = self.global_pooling(x)
        v = self.fc_layers(v)
        # v = self.sigmoid(v)
        return x * v


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_embeddings, embedding_dim):
        super(Encoder, self).__init__()
        self.quantize_conv = nn.Conv2d(in_channels, embedding_dim, kernel_size=1)
        self.conv1 = DepthwiseSeparableConv2d(embedding_dim, out_channels*2, kernel_size=3, stride=1, padding=1)
        self.se1 = SEBlock(out_channels *2)
        self.conv2 = DepthwiseSeparableConv2d(out_channels*2, embedding_dim, kernel_size=3, stride=1, padding=1)
        self.se2 = SEBlock(embedding_dim)
        self.quantize = VectorQuantizer(num_embeddings, embedding_dim)

    def forward(self, x):
        # print('x',x.shape)
        x = self.quantize_conv(x)
        x = self.conv1(x)
        x = self.se1(x)
        x = self.conv2(x)
        x = self.se2(x)
        # print('x',x.shape)
        quantized_inputs, vq_loss ,encoding_one_hot,encoding_inds = self.quantize(x)
        # print('quantized_inputs:',quantized_inputs.shape)

        return quantized_inputs, vq_loss ,encoding_one_hot,encoding_inds


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_embeddings, embedding_dim):
        super(Decoder, self).__init__()
        self.dequantize_conv = nn.Conv2d(embedding_dim, in_channels, kernel_size=1)
        self.conv1 = DepthwiseSeparableConv2d(in_channels, out_channels*2, kernel_size=3, stride=1, padding=1)
        self.se1 = SEBlock(out_channels*2)
        self.conv2 = DepthwiseSeparableConv2d(out_channels*2, out_channels, kernel_size=3, stride=1, padding=1)
        self.se2 = SEBlock(out_channels)


    def forward(self, z_q):
        # z_q = z_q.permute(0, 2, 3, 1).contiguous()
        # print('decoder z:',z_q.shape)
        x = self.dequantize_conv(z_q)
        x = self.conv1(x)
        x = self.se1(x)
        x = self.conv2(x)
        x = self.se2(x)
        return x

class VQVAE(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 embedding_dim: int,
                 num_embeddings: int,
                 hidden_dims: List = None,
                 beta: float = 0.25,
                 img_size: int = 28,
                 **kwargs) -> None:
        super(VQVAE, self).__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.img_size = img_size
        self.beta = beta

        self.encoder = Encoder(self.in_channels, self.out_channels, self.num_embeddings, self.embedding_dim)

        self.vq_layer = VectorQuantizer(num_embeddings,
                                        embedding_dim,
                                        self.beta)


        self.decoder = Decoder(self.in_channels, self.out_channels, self.num_embeddings, self.embedding_dim)

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        # print('input:',input.shape)
        result = self.encoder(input)
        return [result]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder(z)
        return result

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        quantized_inputs, vq_loss ,encoding_one_hot,encoding_inds = self.encode(input)[0]
        return [self.decode(quantized_inputs), input, vq_loss,encoding_one_hot,encoding_inds]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        vq_loss = args[2]

        # print('loss:',recons.shape,input.shape)

        recons_loss = F.mse_loss(recons, input)
        # recons_loss = F.binary_cross_entropy(recons, input, size_average=False)

        loss = recons_loss + vq_loss
        return {'loss': loss,
                'Reconstruction_Loss': recons_loss,
                'VQ_Loss':vq_loss}

    def sample(self,
               num_samples: int,
               current_device: Union[int, str], **kwargs) -> Tensor:
        raise Warning('VQVAE sampler is not implemented.')

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
    def embed(self, indices):
        return self.vq_layer.embed(indices)


class GatedActivation(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x, y = x.chunk(2, dim=1)
        return torch.tanh(x) * torch.sigmoid(y)


class GatedMaskedConv2d(nn.Module):

    def __init__(self, mask_type, hidden_channels, kernel_size, residual, num_classes):
        super().__init__()
        self.mask_type = mask_type
        self.residual = residual

        h = hidden_channels

        self.class_cond_embedding = nn.Embedding(num_classes, h * 2)

        self.vert_stack = nn.Conv2d(
            h, h * 2, kernel_size=(kernel_size, kernel_size), stride=1,
            padding=(kernel_size // 2, kernel_size // 2)
        )
        self.vert_to_horiz = nn.Conv2d(h * 2, h * 2, kernel_size=1, stride=1, padding=0)
        self.horiz_stack = nn.Conv2d(h, h * 2, kernel_size=(1, kernel_size), stride=1,
                                     padding=(0, kernel_size // 2))
        self.horiz_resid = nn.Conv2d(h, h, kernel_size=1, stride=1, padding=0)

        self.gate = GatedActivation()

    def make_causal(self):
        self.vert_stack.weight.data[:, :, -1].zero_()  # mask final row
        self.horiz_stack.weight.data[:, :, :, -1].zero_()  # mask final column

    def forward(self, x_v, x_h, h):
        if self.mask_type == 'A':
            self.make_causal()
        # print('h:',h)
        h = h.long()
        h = self.class_cond_embedding(h)
        # print('x_v:',x_v.shape)
        h_vert = self.vert_stack(x_v)
        # print('h_vert:',h_vert.shape)
        # h_vert = h_vert[:, :, :x_v.size(-1), :]
        out_v = self.gate(h_vert + h.squeeze()[:, :, None, None])

        h_horiz = self.horiz_stack(x_h)
        # print('h_horiz:',h_horiz.shape)
        # h_horiz = h_horiz[:, :, :, :x_h.size(-2)]
        v2h = self.vert_to_horiz(h_vert)

        out = self.gate(v2h  + h.squeeze()[:, :, None, None])
        if self.residual:
            out_h = self.horiz_resid(out) + x_h
        else:
            out_h = self.horiz_resid(out)

        return out_v, out_h


class GatedPixelCNN(nn.Module):
    def __init__(self, input_dim=940, dim=256, n_layers=9, n_classes=32):
        super().__init__()
        self.dim = dim

        # Create embedding layer to embed input
        self.embedding = nn.Embedding(input_dim, dim)

        # Building the PixelCNN layer by layer
        self.layers = nn.ModuleList()

        # Initial block with Mask-A convolution
        # Rest with Mask-B convolutions
        for i in range(n_layers):
            mask_type = 'A' if i == 0 else 'B'
            kernel = 7 if i == 0 else 3
            residual = False if i == 0 else True

            self.layers.append(
                GatedMaskedConv2d(mask_type, dim, kernel, residual, n_classes)
            )

        # Add the output layer
        self.output_conv = nn.Sequential(
            nn.Conv2d(dim, 512, 1),
            nn.ReLU(True),
            nn.Conv2d(512, input_dim, 1)
        )

        # self.apply(weights_init)

    def forward(self, x, label):
        # shp = x.size()
        # print('x:',x.shape)
        x = self.embedding(x.squeeze().long()) # (B, H, W, C)
        x = x.permute(0, 3, 1, 2)  # (B, C, W, W)
        
        x_v, x_h = x, x
        for i, layer in enumerate(self.layers):
            x_v, x_h = layer(x_v, x_h, label)

        return self.output_conv(x_h)

    def sample(self, batch_size, shape, label=None):
        param = next(self.parameters())
        x = torch.zeros(
            (batch_size, *shape),
            dtype=torch.float, device=param.device
        )
        # print(x.shape)
        label = torch.zeros(batch_size,device=param.device)
        # print(label.shape)

        
        for i in range(shape[0]):
            for j in range(shape[1]):
                for z in range(shape[2]):
                    logits = self.forward(x, label)
                    # print(logits.shape)
                    probs = F.softmax(logits[:, i, j, z], -1)
                    x.data[:,i, j,z].copy_(
                        probs.multinomial(1).squeeze().data
                    )
        return x


class PixelCNNVQVAE(nn.Module):

    def __init__(self, pixelcnn, vqvae,latent_height, latent_width, channle):
        super().__init__()
        self.pixelcnn = pixelcnn
        self.vqvae = vqvae
        self.channle = channle
        self.latent_height = latent_height
        self.latent_width = latent_width

    def sample_prior(self, batch_size, label=None):
        indices = self.pixelcnn.sample(batch_size, (self.latent_height, self.latent_width,self.channle), label).squeeze(dim=1)
        print('indices:',indices.shape)
        self.vqvae.eval()
        with torch.no_grad():
            quantized = self.pixelcnn.embedding(indices).permute(0, 3, 1, 2)

        return quantized, indices

    def sample(self, batch_size, label=None):
        quantized, indices = self.sample_prior(batch_size, label)
        with torch.no_grad():
            x_recon = self.vqvae.decode(quantized)
        return x_recon, quantized, indices