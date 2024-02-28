"""
date:
author:
"""
import warnings
warnings.filterwarnings("ignore")
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


class ResBlock(nn.Module):

    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(True),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        return x + self.block(x)


class VQ(nn.Module):

    def __init__(self, num_embeddings, embedding_size, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_size = embedding_size
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(num_embeddings, embedding_size)
        self.embedding.weight.data.uniform_(-1. / num_embeddings, 1. / num_embeddings)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()  # from BCHW to BHWC
        x_flat = x.view(-1, self.embedding_size)

        w = self.embedding.weight
        distances = torch.sum(x_flat ** 2, dim=1, keepdim=True) + torch.sum(w ** 2, dim=1) - 2 * (x_flat @ w.T)
        indices_flat = torch.argmin(distances, dim=1, keepdim=True)
        quantized_flat = self.embed(indices_flat)

        quantized = quantized_flat.view(x.shape)
        indices = indices_flat.view(*x.shape[:3]).unsqueeze(dim=1)  # BHW to BCHW

        if self.training:
            e_latent_loss = F.mse_loss(quantized.detach(), x)
            q_latent_loss = F.mse_loss(quantized, x.detach())
            loss = q_latent_loss + self.commitment_cost * e_latent_loss

            quantized = x + (quantized - x).detach()
        else:
            loss = 0.

        quantized = quantized.permute(0, 3, 1, 2).contiguous()  # from BHWC to BCHW

        return quantized, indices, loss

    def embed(self, indices):
        quantized = self.embedding(indices.long())
        return quantized


class VQVAE(nn.Module):

    def __init__(self, in_channels, num_embeddings, embedding_size=32, res_hidden_channels=32, commitment_cost=0.25):
        super().__init__()
        self.in_channels = in_channels
        self.num_embeddings = num_embeddings
        self.embedding_size = embedding_size

        h = embedding_size
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, h, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(h),
            nn.ReLU(inplace=True),
            nn.Conv2d(h, h, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(h),
            ResBlock(h, res_hidden_channels),
            ResBlock(h, res_hidden_channels),
            ResBlock(h, res_hidden_channels)
        )

        self.vq = VQ(num_embeddings, embedding_size, commitment_cost)

        self.decoder = nn.Sequential(
            ResBlock(h, res_hidden_channels),
            ResBlock(h, res_hidden_channels),
            ResBlock(h, res_hidden_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(h, h, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(h),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(h, in_channels, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x):
        z = self.encode(x)
        quantized, indices, vq_loss = self.quantize(z) # 初步认为 quantized为Z帽 indices为码矢
        x_recon = self.decode(quantized)
        return x_recon,x, quantized, indices, vq_loss

    def encode(self, x):
        z = self.encoder(x)
        return z

    def quantize(self, z):
        quantized, indices, vq_loss = self.vq(z)
        return quantized, indices, vq_loss

    def decode(self, quantized):
        x_recon = self.decoder(quantized)
        return x_recon
    
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

    def embed(self, indices):
        return self.vq.embed(indices)

def generate_samples(images, model):
  with torch.no_grad():
    images = images.to(device)
    x_recon, _, _, _ = model(images)
  return x_recon


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
            h, h * 2, kernel_size=(kernel_size // 2 + 1, kernel_size), stride=1,
            padding=(kernel_size // 2, kernel_size // 2)
        )
        self.vert_to_horiz = nn.Conv2d(h * 2, h * 2, kernel_size=1, stride=1, padding=0)
        self.horiz_stack = nn.Conv2d(h, h * 2, kernel_size=(1, kernel_size // 2 + 1), stride=1,
                                     padding=(0, kernel_size // 2))
        self.horiz_resid = nn.Conv2d(h, h, kernel_size=1, stride=1, padding=0)

        self.gate = GatedActivation()

    def make_causal(self):
        self.vert_stack.weight.data[:, :, -1].zero_()  # mask final row
        self.horiz_stack.weight.data[:, :, :, -1].zero_()  # mask final column

    def forward(self, x_v, x_h, h):
        if self.mask_type == 'A':
            self.make_causal()
        h = h.squeeze().long()
        h = self.class_cond_embedding(h)
        h_vert = self.vert_stack(x_v)
        h_vert = h_vert[:, :, :x_v.size(-1), :]
        # print('vert:',h_vert.shape)
        # print(h.shape)
        out_v = self.gate(h_vert + h[:, :, None, None])

        h_horiz = self.horiz_stack(x_h)
        h_horiz = h_horiz[:, :, :, :x_h.size(-2)]
        # print('horiz:',h_horiz.shape)
        v2h = self.vert_to_horiz(h_vert)
        # print('v2h:', v2h.shape)
        out = self.gate(v2h + h_horiz + h[:, :, None, None])
        if self.residual:
            out_h = self.horiz_resid(out) + x_h
        else:
            out_h = self.horiz_resid(out)

        return out_v, out_h


class GatedPixelCNN(nn.Module):

    def __init__(self, num_classes=32, in_channels=32, hidden_channels=32, output_channels=64, num_layers=5):
        super().__init__()
        self.num_classes = num_classes

        self.embedding = nn.Embedding(in_channels, hidden_channels)

        self.mask_a = GatedMaskedConv2d('A', hidden_channels, kernel_size=7,
                                        residual=False, num_classes=num_classes)
        self.mask_bs = nn.ModuleList([
            GatedMaskedConv2d('B', hidden_channels, kernel_size=3,
                              residual=True, num_classes=num_classes) for _ in range(num_layers - 1)])

        self.output_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, output_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, in_channels, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x, y):
        x = self.embedding(x.squeeze().long())  # (B, H, W, C)
        # print('x:',x.shape)
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)

        x_v, x_h = self.mask_a(x, x, y)
        for mask_b in self.mask_bs:
            x_v, x_h = mask_b(x_v, x_h, y)
        # print('for x_h:',x_h.shape)
        # print('self.output_conv(x_h):',self.output_conv(x_h).shape)

        return self.output_conv(x_h)

    def sample(self, batch_size, shape, label=None):
        self.eval()

        x = torch.zeros((batch_size, *shape), dtype=torch.float32, device=device)
        if label is None:
            label = torch.randint(self.num_classes, (batch_size,), dtype=torch.long, device=device)

        with torch.no_grad():
            for i in range(shape[0]):
                for j in range(shape[1]):
                    logits = self.forward(x, label)
                    dist = Categorical(logits=logits[:, :, i, j])
                    x[:, i, j] = dist.sample()
        return x

class PixelCNNVQVAE(nn.Module):

    def __init__(self, pixelcnn, vqvae, latent_height, latent_width):
        super().__init__()
        self.pixelcnn = pixelcnn
        self.vqvae = vqvae
        self.latent_height = latent_height
        self.latent_width = latent_width

    def sample_prior(self, batch_size, label=None):
        indices = self.pixelcnn.sample(batch_size, (self.latent_height, self.latent_width), label).squeeze(dim=1)

        self.vqvae.eval()
        with torch.no_grad():
            quantized = self.vqvae.embed(indices)
        # print('print(indices)1:',indices)

        return quantized, indices

    def sample(self, batch_size, label=None):
        quantized, indices = self.sample_prior(batch_size, label)
        # print('print(indices)2:',indices)
        with torch.no_grad():
            x_recon = self.vqvae.decode(quantized) #quantized 为 zq

            # print(x_recon.shape)
        return x_recon, quantized, indices