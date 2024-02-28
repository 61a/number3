import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from torchvision import transforms,datasets
from torchvision.transforms import ToPILImage
from torch.utils.data import DataLoader,Dataset
from PIL import Image
from model.VQVAE.transformer import VQVAETransformer

class ImagePaths(Dataset):
    def __init__(self, path, size=None):
        self.size = size

        self.images = [os.path.join(path, file) for file in os.listdir(path)]
        self._length = len(self.images)

        self.transforms = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            # transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        # if not image.mode == "RGB":
        #     image = image.convert("RGB")
        # image = np.array(image).astype(np.uint8)
        image = self.transforms(image)
        # image = (image / 127.5 - 1.0).astype(np.float32)
        # image = image.transpose(2, 0, 1)
        return image

    def __getitem__(self, i):
        example = self.preprocess_image(self.images[i])
        to_pil = ToPILImage()
        image = to_pil(example)
        # example = self.transforms(example)
        return example

class TrainTransformer:
    def __init__(self, args):
        if args.dataset == 'HTRU':
            # HTRU_net = torch.load(os.path.join("./vqvae_gpt/cifar10_gpt_checkpoints", f"transformer_best.pt"))
            self.model = VQVAETransformer(args)
            # self.model.load_state_dict(HTRU_net['state_dict'])
            self.model.to(device=args.device) 
        elif args.dataset == 'cifar10':
            self.model = VQVAETransformer(args).to(device=args.device)

        self.optim = self.configure_optimizers()

        self.prepare_training(args)

        self.train(args)

    def prepare_training(self,args):
        os.makedirs("./vqvae_gpt/{}_gpt_results_{}".format(args.dataset,args.Versions), exist_ok=True)
        os.makedirs("./vqvae_gpt/{}_gpt_checkpoints_{}".format(args.dataset,args.Versions), exist_ok=True)

    def configure_optimizers(self):
        decay, no_decay = set(), set()
        whitelist_weight_modules = (nn.Linear, )
        blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)

        for mn, m in self.model.transformer.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn

                if pn.endswith("bias"):
                    no_decay.add(fpn)

                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)

                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        # no_decay.add("pos_emb")

        param_dict = {pn: p for pn, p in self.model.transformer.named_parameters()}

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=4.5e-06, betas=(0.9, 0.95))
        return optimizer
    def load_data(self,args):
        if args.dataset == 'HTRU':
            train_data = ImagePaths(args.dataset_path, size=32)
            train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False,drop_last=True)
        elif args.dataset == 'cifar10':
            transform = transforms.Compose([
                        transforms.Resize((args.image_size, args.image_size)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                    ])
            train_dataset = datasets.CIFAR10(root=args.dataset_path, train=True, download=True, transform=transform)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,drop_last=True)

        return train_loader

    def train(self, args):
        train_dataset = self.load_data(args)
        best_loss = 100
        gpt_loss = []
        for epoch in range(args.epochs):
            with tqdm(range(len(train_dataset))) as pbar:
                for i, imgs in zip(pbar, train_dataset):
                    self.optim.zero_grad()
                    if args.dataset == 'HTRU':
                        imgs = imgs.to(device=args.device)
                    elif args.dataset == 'cifar10':
                        imgs = imgs[0].to(device=args.device)
                    logits, targets = self.model(imgs)
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                    loss.backward()
                    self.optim.step()
                    pbar.set_postfix(Epoch = epoch,Transformer_Loss=np.round(loss.cpu().detach().numpy().item(), 4))
                    pbar.update(0)
            log, sampled_imgs = self.model.log_images(imgs[0][None])
            if epoch % 50 == 0:
                vutils.save_image(sampled_imgs, os.path.join("./vqvae_gpt/{}_gpt_results_{}".format(args.dataset,args.Versions), f"transformer_{epoch}.jpg"), nrow=4)
                torch.save(self.model.state_dict(), 
                           os.path.join("./vqvae_gpt/{}_gpt_checkpoints_{}".format(args.dataset,args.Versions), f"transformer_{epoch}.pt"))
            if loss.cpu().detach().numpy().item() < best_loss and epoch > 20:
                best_loss = loss.cpu().detach().numpy().item()
                torch.save({'state_dict': self.model.state_dict(),'best_epoch': epoch},
                            os.path.join("./vqvae_gpt/{}_gpt_checkpoints_{}".format(args.dataset,args.Versions), f"transformer_best.pt"))
            gpt_loss.append(loss.cpu().detach().numpy().item())
            np.save(os.path.join("./vqvae_gpt/{}_gpt_checkpoints_{}".format(args.dataset,args.Versions), f"gpt_loss.npy"),gpt_loss)
       

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQVAE")
    parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension n_z.')
    parser.add_argument('--image-size', type=int, default=32, help='Image height and width.)')
    parser.add_argument('--num-codebook-vectors', type=int, default=2048, help='Number of codebook vectors.')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar.')
    parser.add_argument('--image-channels', type=int, default=3, help='Number of channels of images.')
    parser.add_argument('--dataset', type=str, default='HTRU', help='dataset name.')
    parser.add_argument('--dataset-path', type=str, default='./data', help='Path to data.')
    parser.add_argument('--checkpoint-path', type=str, default='./checkpoints/last_ckpt.pt', help='Path to checkpoint.')
    parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on')
    parser.add_argument('--batch-size', type=int, default=64, help='Input batch size for training.')
    parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs to train.')
    parser.add_argument('--learning-rate', type=float, default=2.25e-05, help='Learning rate.')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param.')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta param.')
    parser.add_argument('--l2-loss-factor', type=float, default=1., help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perceptual-loss-factor', type=float, default=1., help='Weighting factor for perceptual loss.')

    parser.add_argument('--pkeep', type=float, default=0.5, help='Percentage for how much latent codes to keep.')
    parser.add_argument('--sos-token', type=int, default=0, help='Start of Sentence token.')
    parser.add_argument('--Versions', type=str, default='HTRU-FAST-profile-sub', help='training Versions')
    # parser.add_argument('--Versions', type=str, default='cifar10', help='training Versions')

    args = parser.parse_args()
    if args.dataset == 'HTRU':
        args.dataset_path = '/aidata/Ly61/number3/HTRU-FAST-profile-submerge/pulsar'
        args.checkpoint_path = "./vqvae_gpt/{}_vqvae_checkpoints_{}/vqvae_epoch_best.pt".format(args.dataset,args.Versions)
    elif args.dataset == 'cifar10':
        args.dataset_path = './data'
        args.checkpoint_path = "./vqvae_gpt/{}_vqvae_checkpoints_{}/vqvae_epoch_best.pt".format(args.dataset,args.Versions)
        args.batch_size = 128

    train_transformer = TrainTransformer(args)


