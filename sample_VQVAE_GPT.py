import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import argparse
import torch
from torchvision import utils as vutils
from model.VQVAE.transformer import VQVAETransformer,VQVAE
from tqdm import tqdm


parser = argparse.ArgumentParser(description="VQVAE")
parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension n_z.')
parser.add_argument('--image-size', type=int, default=32, help='Image height and width.)')
parser.add_argument('--num-codebook-vectors', type=int, default=2048, help='Number of codebook vectors.')
parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar.')
parser.add_argument('--image-channels', type=int, default=3, help='Number of channels of images.')
parser.add_argument('--dataset', type=str, default='HTRU', help='dataset name.')
parser.add_argument('--dataset-path', type=str, default='./data', help='Path to data.')
parser.add_argument('--checkpoint-path', type=str, default='./checkpoints/last_ckpt.pt',
                    help='Path to checkpoint.')
parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on')
parser.add_argument('--batch-size', type=int, default=20, help='Input batch size for training.')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--learning-rate', type=float, default=2.25e-05, help='Learning rate.')
parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param.')
parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta param.')
parser.add_argument('--disc-start', type=int, default=10000, help='When to start the discriminator.')
parser.add_argument('--disc-factor', type=float, default=1., help='Weighting factor for the Discriminator.')
parser.add_argument('--l2-loss-factor', type=float, default=1.,
                    help='Weighting factor for reconstruction loss.')
parser.add_argument('--perceptual-loss-factor', type=float, default=1.,
                    help='Weighting factor for perceptual loss.')

parser.add_argument('--pkeep', type=float, default=0.5, help='Percentage for how much latent codes to keep.')
parser.add_argument('--sos-token', type=int, default=0, help='Start of Sentence token.')
parser.add_argument('--Versions', type=str, default='HTRU-FAST-profile-sub', help='training Versions')

args = parser.parse_args()

###########add
# n = 5
# args.dataset_path = '../HTRU-pulsar/pulsar'
# args.checkpoint_path = "./vqvae_gpt/{}_vqvae_checkpoints_{}/vqvae_epoch_best.pt".format(args.dataset,args.Versions)
# model = VQVAE(args)
# model.load_checkpoint(args.checkpoint_path)
# model = model.eval()
# for i in tqdm(range(n)):
#     sos_tokens = torch.rand((5,3, 32, 32)).float().cuda()
#     decoded_images,_,_ = model(sos_tokens)
#     sampled_imgs = decoded_images
#     vutils.save_image(sampled_imgs, os.path.join("./vqvae_gpt/{}_vqvae_results/".format(args.dataset,args.Versions),  f"ceshi_{i}.jpg"), nrow=1)

#######add

def prepare_training(args):
        os.makedirs(os.path.join("./vqvae_gpt/{}_VQVAE_gpt_sample/VQVAE_pulsar_{}".format(args.dataset,args.Versions), "sample-{}".format(args.Versions)), exist_ok=True)
        os.makedirs(os.path.join("./vqvae_gpt/{}_VQVAE_gpt_sample/VQVAE_pulsar_{}".format(args.dataset,5), "sample-{}".format(5)), exist_ok=True)

prepare_training(args)
if args.dataset == 'HTRU':
    # args.dataset_path = '../HTRU-subpulsar/pulsar'
    args.dataset_path = '/aidata/Ly61/number3/HTRU-FAST-profile-submerge/pulsar'
    args.checkpoint_path = "./vqvae_gpt/{}_vqvae_checkpoints_{}/vqvae_epoch_best.pt".format(args.dataset,args.Versions)
elif args.dataset == 'cifar10':
    args.dataset_path = './data'
    args.checkpoint_path = "./vqvae_gpt/{}_vqvae_checkpoints_{}/vqvae_epoch_best.pt".format(args.dataset,args.Versions)
    args.batch_size = 128

n = 50000
transformer = VQVAETransformer(args)
transformer.load_state_dict(torch.load(os.path.join("./vqvae_gpt/{}_gpt_checkpoints_{}".format(args.dataset,args.Versions), f"transformer_best.pt"))['state_dict'])
transformer.to(device=args.device) 
print("Loaded state dict of Transformer")
os.makedirs("./vqvae_gpt/{}_VQVAE_gpt_sample/VQVAE_pulsar_{}/sample-{}".format(args.dataset,args.Versions,args.Versions), exist_ok=True)
for i in tqdm(range(n)):
    start_indices = torch.zeros((1, 0)).long().to("cuda")
    sos_tokens = torch.ones(start_indices.shape[0], 1) * 0
    sos_tokens = sos_tokens.long().to("cuda")
    sample_indices = transformer.sample(start_indices, sos_tokens, steps=4)
    sampled_imgs = transformer.z_to_image(sample_indices)
    vutils.save_image(sampled_imgs, os.path.join("./vqvae_gpt/{}_VQVAE_gpt_sample/VQVAE_pulsar_{}".format(args.dataset,args.Versions), "sample-{}".format(args.Versions), f"sample_{i}.png"), nrow=1)
    vutils.save_image(sampled_imgs, os.path.join("./vqvae_gpt/{}_VQVAE_gpt_sample/VQVAE_pulsar_{}".format(args.dataset,5), "sample-{}".format(5), f"sample_{i}.png"), nrow=1)
