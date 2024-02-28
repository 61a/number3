import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import utils as vutils
from torchvision import transforms,datasets
from torchvision.transforms import ToPILImage
from torch.utils.data import DataLoader,Dataset
# from model.VQVAE.lpips import LPIPS
from model.VQVAE.vqvae import VQVAE
from PIL import Image
import pandas as pd



# class ImagePaths(Dataset):
#     def __init__(self, path, size=None):
#         self.size = size
#         self.root_dir = path
#         self.num_to_select = 800

#         # self.images = [os.path.join(path, file) for file in os.listdir(path)]

#         # 所有图像路径列表
#         self.images = []
#         # 遍历root_dir下的每个子目录
#         for subdir in os.listdir(self.root_dir):
#             subdir_path = os.path.join(self.root_dir, subdir)
#             if os.path.isdir(subdir_path):
#                 # 获取子目录中所有图像的路径
#                 subdir_images = [
#                     os.path.join(subdir_path, file) for file in os.listdir(subdir_path) 
#                     if os.path.isfile(os.path.join(subdir_path, file))]
#                 # 如果需要选择特定数量的图像
#                 if self.num_to_select and (subdir == 'pulsar' or subdir == 'unpulsar'):  # 特别设置仅针对'unpulsar'文件夹
#                     subdir_images = subdir_images[:self.num_to_select]
#                 # 添加到主列表
#                 self.images.extend(subdir_images)



#         self._length = len(self.images)

#         self.transforms = transforms.Compose([
#             transforms.Resize((self.size, self.size)),
#             # transforms.Grayscale(num_output_channels=1),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
#         ])

#     def __len__(self):
#         return self._length

#     def preprocess_image(self, image_path):
#         image = Image.open(image_path)
#         # if not image.mode == "RGB":
#         #     image = image.convert("RGB")
#         # image = np.array(image).astype(np.uint8)
#         image = self.transforms(image)
#         # image = (image / 127.5 - 1.0).astype(np.float32)
#         # image = image.transpose(2, 0, 1)
#         return image

#     def __getitem__(self, i):
#         example = self.preprocess_image(self.images[i])
#         # to_pil = ToPILImage()
#         # image = to_pil(example)
#         # example = self.transforms(example)
#         return example

class ImagePaths(Dataset):
    def __init__(self, csv_file,size=None):
        self.size = size
        # 读取CSV文件
        self.dataframe = pd.read_csv(csv_file,sep='\t')
        # 假设图像路径保存在列'image'中
        self.image_paths = self.dataframe['image'].tolist()

        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, i):
        image_path = self.image_paths[i]
        image = Image.open(image_path)
        label = self.dataframe.iloc[i, 3]
        if self.transform:
            image = self.transform(image)
        return image

class TrainVQVAE:
    def __init__(self, args):
        self.vqvae = VQVAE(args).to(device=args.device)
        self.opt_vq = self.configure_optimizers(args)
        self.prepare_training(args)
        self.train(args)

    def prepare_training(self,args):
        os.makedirs("/aidata/Ly61/xian/vqvae_gpt/{}_vqvae_results_{}".format(args.dataset,args.Versions), exist_ok=True)
        os.makedirs("/aidata/Ly61/xian/vqvae_gpt/{}_vqvae_checkpoints_{}".format(args.dataset,args.Versions), exist_ok=True)

    def configure_optimizers(self, args):
        lr = args.learning_rate
        opt_vq = torch.optim.Adam(
            list(self.vqvae.encoder.parameters()) +
            list(self.vqvae.decoder.parameters()) +
            list(self.vqvae.codebook.parameters()) +
            list(self.vqvae.quant_conv.parameters()) +
            list(self.vqvae.post_quant_conv.parameters()),
            lr=lr, eps=1e-08, betas=(args.beta1, args.beta2)
        )
        
        return opt_vq
    
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
        train_loader = self.load_data(args)
        best_vq_loss = 100
        VQ_loss = []
        ### EMA update loss factor
        ema_perceptual_loss = None
        ema_rec_loss = None
        ema_perceptual_rec_loss = None
        ema_q_loss = None
        alpha = 0.9
        for epoch in range(args.epochs):
            with tqdm(range(len(train_loader))) as pbar:
                for i, imgs in zip(pbar, train_loader):
                    # print(imgs)
                    if args.dataset == 'HTRU':
                        imgs = imgs.to(device=args.device)
                    elif args.dataset == 'cifar10':
                        imgs = imgs[0].to(device=args.device)
                    decoded_images, codecook_index, q_loss = self.vqvae(imgs)

                    perceptual_loss = F.mse_loss(imgs, decoded_images)
                    rec_loss = torch.abs(imgs - decoded_images).mean()

                    # if ema_perceptual_loss is None:
                    #     ema_perceptual_loss = perceptual_loss.item()
                    # else:
                    #     ema_perceptual_loss = alpha * ema_perceptual_loss + (1-alpha) * perceptual_loss.item()
                    
                    # if ema_rec_loss is None:
                    #     ema_rec_loss = rec_loss.item()
                    # else:
                    #     ema_rec_loss = alpha * ema_rec_loss + (1-alpha) * rec_loss.item()

                    # args.perceptual_loss_factor = ema_rec_loss / (ema_rec_loss + ema_perceptual_loss)
                    # args.rec_loss_factor = ema_perceptual_loss / (ema_rec_loss + ema_perceptual_loss)


                    perceptual_rec_loss = args.perceptual_loss_factor * perceptual_loss + args.rec_loss_factor * rec_loss
                    # perceptual_rec_loss = perceptual_rec_loss.mean()

                    # if ema_perceptual_rec_loss is None:
                    #     ema_perceptual_rec_loss = perceptual_rec_loss.item()
                    # else:
                    #     ema_perceptual_rec_loss = alpha * ema_perceptual_rec_loss + (1-alpha) * perceptual_rec_loss.item()
                    
                    # if ema_q_loss is None:
                    #     ema_q_loss = q_loss.item()
                    # else:
                    #     ema_q_loss = alpha * ema_q_loss + (1-alpha) * q_loss.item()

                    # args.perceptual_rec_loss_factor = ema_q_loss / (ema_q_loss + ema_perceptual_rec_loss)
                    # args.q_loss_factor = ema_perceptual_rec_loss / (ema_q_loss + ema_perceptual_rec_loss)
                    

                    vq_loss = args.perceptual_rec_loss_factor * perceptual_rec_loss + args.q_loss_factor * q_loss


                    self.opt_vq.zero_grad()
                    torch.autograd.set_detect_anomaly(True)
                    vq_loss.backward(retain_graph=True)


                    self.opt_vq.step()

                    if (i+1) % len(train_loader) == 0 and epoch % 20 == 0:
                        with torch.no_grad():
                            real_fake_images = torch.cat((imgs[:20], decoded_images.add(1).mul(0.5)[:20]))
                            vutils.save_image(real_fake_images, os.path.join("/aidata/Ly61/xian/vqvae_gpt/{}_vqvae_results_{}".format(args.dataset,args.Versions), f"{epoch}_{i}.jpg"), nrow=4)
                            
                    
                    pbar.set_postfix(Epoch = epoch,
                        VQ_Loss=np.round(vq_loss.cpu().detach().numpy().item(), 5)
                    )
                    pbar.update(0)
                
                if epoch % 50 == 0:
                    torch.save(self.vqvae.state_dict(), os.path.join("/aidata/Ly61/xian/vqvae_gpt/{}_vqvae_checkpoints_{}".format(args.dataset,args.Versions), f"vqvae_epoch_{epoch}.pt"))
                if vq_loss.cpu().detach().numpy().item() < best_vq_loss and epoch > 20:
                    best_vq_loss = vq_loss.cpu().detach().numpy().item()
                    torch.save({'state_dict': self.vqvae.state_dict(),'best_epoch': epoch},
                                os.path.join("/aidata/Ly61/xian/vqvae_gpt/{}_vqvae_checkpoints_{}".format(args.dataset,args.Versions), f"vqvae_epoch_best.pt"))
            VQ_loss.append(vq_loss.cpu().detach().numpy().item())
            # np.save(os.path.join("/aidata/Ly61/xian/vqvae_gpt/{}_vqvsae_checkpoints_{}".format(args.dataset,args.Versions), f"VQloss.npy"),VQ_loss)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQVAE")
    parser.add_argument('--latent-dim', type=int, default=2048, help='Latent dimension n_z (default: 256)')
    parser.add_argument('--image-size', type=int, default=1024, help='Image height and width (default: 32)')
    parser.add_argument('--num-codebook-vectors', type=int, default=2048, help='Number of codebook vectors (default: 256)')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar (default: 0.25)')
    parser.add_argument('--image-channels', type=int, default=1, help='Number of channels of images (default: 3)')
    parser.add_argument('--dataset', type=str, default='HTRU', help='dataset name.')
    parser.add_argument('--dataset-path', type=str, default='/data', help='Path to data (default: /data)')
    parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on')
    parser.add_argument('--batch-size', type=int, default=256, help='Input batch size for training (default: 32)')
    parser.add_argument('--test_batch-size', type=int, default=256, help='Input batch size for test (default: 32)')
    parser.add_argument('--epochs', type=int, default=400, help='Number of epochs to train (default: 50)')
    parser.add_argument('--learning-rate', type=float, default=2.25e-05, help='Learning rate (default: 0.0002)')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param (default: 0.0)')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta param (default: 0.999)')
    parser.add_argument('--rec-loss-factor', type=float, default=0.5, help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perceptual-loss-factor', type=float, default=0.5, help='Weighting factor for perceptual loss.')
    parser.add_argument('--perceptual-rec-loss-factor', type=float, default=0.5, help='Weighting factor for perceptual reconstruction loss.')
    parser.add_argument('--q-loss-factor', type=float, default=0.5, help='Weighting factor for q loss.')
    parser.add_argument('--Versions', type=str, default='Pneumonia', help='training Versions')
    # parser.add_argument('--Versions', type=str, default='cifar10', help='training Versions')



    args = parser.parse_args()
    if args.dataset == 'HTRU':
        # args.dataset_path = '/aidata/Ly61/number3/HTRU-FAST-profile-submerge/pulsar'
        # args.dataset_path = '/aidata/Ly61/number3/FAST-profile-submerge'
        args.dataset_path = '/aidata/Ly61/xian/MIMIC/train_AP.csv'
    elif args.dataset == 'cifar10':
        args.dataset_path = './data'
        args.batch_size = 128

    train_vqvae = TrainVQVAE(args)
