import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torchvision
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torchvision.utils import save_image
import cv2 
import matplotlib.pyplot as plt 
from copy import deepcopy 
import os 
from tqdm import tqdm 

from utils_transformer import * 


# class for transformer input embedding
class Transformer_Embeddings(nn.Module):
    def __init__(self, x_dim, max_seq_len, d_model, dropout, device):
        super().__init__()
        self.x_emb = nn.Linear(x_dim, d_model, bias=False)
        self.pos_emb = nn.Parameter(torch.randn(max_seq_len, d_model)) 
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.max_seq_len = max_seq_len
        self.device = device
    def forward(self, x): # s.shape: [batch_size, max_seq_len, x_dim]
        batch_size, input_seq_len = x.shape[0], x.shape[1]
        x_emb = self.x_emb(x) # x_emb.shape: [batch_size, max_seq_len, d_model]
        # add positional embeddings
        pos_emb = self.pos_emb
        pos_emb = pos_emb.unsqueeze(0) # pos_emb.shape: [1, max_seq_len, d_model]
        pos_emb = pos_emb.expand(batch_size, -1, -1) # pos_emb.shape: [batch_size, max_seq_len, d_model]
        final_emb = self.dropout( self.norm(x_emb + pos_emb) )
        return final_emb
    

# class implementing Transformer 
# NOTE that since our goal here is just compression, this is a non-causal encoder-only transformer
class Transformer(nn.Module):
    def __init__(self, embedder, encoder, d_model, out_dim):
        super().__init__()
        self.embedder = embedder 
        self.encoder = encoder
        self.out_proj = nn.Linear(d_model, out_dim)

    def forward(self, x): # x.shape: [batch_size, max_seq_len, x_dim]
        x_emb = self.embedder(x) # x_emb.shape: [batch_size, max_seq_len, d_model]
        encoder_out = self.encoder(x_emb) # encoder_out.shape: [batch_size, max_seq_len, d_model]
        final_out = self.out_proj(encoder_out) # final_out.shape: [batch_size, max_seq_len, out_dim]
        return final_out
    

# caller function to instantiate the transformer, using the defined hyperparams as input
def init_transformer(x_dim, max_seq_len, d_model, d_k, d_v, n_heads, n_layers, d_ff, dropout, out_dim, device):
    attn = MultiHeadAttention(n_heads, d_model, d_k, d_v, dropout) # multi head attention block
    ff = FeedForward(d_model, d_ff, dropout) # feed forward block for each encoder / decoder block
    embedder = Transformer_Embeddings(x_dim, max_seq_len, d_model, dropout, device) # embedder block to obtain sequence of embeddings from sequence of input tokens
    encoder_layer = EncoderLayer(deepcopy(attn), deepcopy(ff), d_model, dropout) # single encoder layer
    encoder = Encoder(encoder_layer, n_layers, d_model) # encoder = stacked encoder layers
    model = Transformer(embedder, encoder, d_model, out_dim) # a non-causal encoder-only transfomer
    # initialize params - Xavier initialization
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


class VAE_Transformer(nn.Module):
    def __init__(self, latent_dim, encoder_transformer, decoder_transformer, max_seq_len, device):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_latents = max_seq_len 
        self.encoder = encoder_transformer 
        self.decoder = decoder_transformer 
        self.device = device 

    def encode(self, x): # [b, max_seq_len, x_dim] NOTE that max_seq_len = num_latents per x
        x = self.encoder(x) # [b, max_seq_len, latent_dim * 2]
        mu, logvar = x.chunk(2, dim=-1) # mu.shape: [b, max_seq_len, latent_dim]
        return mu, logvar 

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.FloatTensor(std.size()).normal_().to(self.device)
        return mu + std * eps 

    def decode(self, z): # z.shape: [b, max_seq_len, latent_dim]
        x = self.decoder(z) # [b, max_seq_len, x_dim]
        # x = torch.tanh(z) # project all pixel values to be in range [-1, 1] since training imgs are in this range - NOTE this is not necessary and dilutes the loss signal
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x = self.decode(z)
        return x, mu, logvar
    

# utility function to convert img to sequence of patches (used to process img data)
def img_to_patch_seq(img, patch_size, num_latents):
    b,c,h,w = img.shape
    p = patch_size 
    assert (h % p == 0) and (w % p == 0)
    seq_len = (h * w) // (p * p)
    assert num_latents == seq_len 
    n_rows, n_cols = h//p, w//p
    patch_seq = []
    for row in range(n_rows):
        for col in range(n_cols):
            i = row * p
            j = col * p
            patch = img[:, :, i:i+p, j:j+p]
            patch = torch.flatten(patch, start_dim=1, end_dim=-1) # patch.shape: [b, patch_dim]
            patch_seq.append(patch)
    patch_seq = torch.stack(patch_seq, dim=0) # patch_seq.shape: [seq_len, b, patch_dim]
    patch_seq = patch_seq.transpose(0, 1) # patch_seq.shape: [b, seq_len, patch_dim]
    # assert seq_len == patch_seq.shape[1]
    # assert patch_dim == patch_seq.shape[-1]
    return patch_seq

# utility function to convert sequence of patches to an img (used to process img data)
def patch_seq_to_img(x, patch_size, channels): # x.shape: [batch_size, max_seq_len, patch_dim]
    batch_size, seq_len, patch_dim = x.shape[0], x.shape[1], x.shape[2]
    p = patch_size 
    x = x.permute(1, 0, 2) # x.shape: [seq_len, batch_size, patch_dim]
    x = x.reshape(seq_len, batch_size, channels, p*p)
    x = x.reshape(seq_len, batch_size, channels, p, p)
    nrows = int(math.sqrt(seq_len))
    ncols = nrows
    for i in range(nrows):
        for j in range(ncols):
            if j == 0:
                row = x[i * nrows + j]
            else:
                row = torch.cat((row, x[i * nrows + j]), dim=-1)
        if i == 0:
            imgs = row # row.shape: [batch_size, 3, p, w]
        else:
            imgs = torch.cat((imgs, row), dim=-2) # imgs.shape: [batch_size, 3, h, w]
    # imgs = imgs.permute(0, 2, 3, 1) # imgs.shape: [batch_size, h, w, 3]
    return imgs 

    
# convert flat tensor to img
def to_img(x):
    x = x.clamp(-1, 1) # clamp img to be strictly in [-1, 1]
    x = 0.5 * x + 0.5 # transform img from range [-1, 1] -> [0, 1]
    x = x.permute(0,2,3,1)
    return x

# function to generate img from VAE 
def generate_img(vae, latent_size, device):
    # z = torch.FloatTensor(latent_size).uniform_().to(device)
    z = torch.FloatTensor(latent_size).normal_().to(device)
    x_sampled = vae.decode(z.unsqueeze(0))
    # convert patch sequence to img 
    x_sampled = patch_seq_to_img(x_sampled, patch_size, img_channels)
    img_sampled = to_img(x_sampled)
    return img_sampled 

# VAE loss 
def loss_function(recon_x, x, mu, logvar):
    reconstruction_loss = reconstruction_loss_function(recon_x, x)
    KLdiv_element = 1 + logvar - mu.pow(2) - torch.exp(logvar) # KLdiv_element.shape: [b, 48]
    # KLdiv_loss = -0.5 * torch.sum(KLdiv_element, dim=-1)
    # KLdiv_loss = KLdiv_loss.mean()
    KLdiv_loss = -0.5 * torch.sum(KLdiv_element)
    return reconstruction_loss + KLdiv_loss

# function to save a test img and its reconstructed img 
def save_image_recon(x, x_r, save_path):
    concat_img = torch.cat([x, x_r], dim=1)
    concat_img = concat_img.detach().cpu().numpy()
    concat_img = np.uint8( concat_img * 255 )
    # bgr to rgb 
    concat_img = concat_img[:, :, ::-1]
    cv2.imwrite(save_path, concat_img)

# function to save a generated img 
def save_image_gen(x, save_path):
    x = x.detach().cpu().numpy()
    x = np.uint8( x * 255 )
    # bgr to rgb 
    x = x[:, :, ::-1]
    cv2.imwrite(save_path, x)

# utility function to load model weights from checkpoint - loads to the device passed as 'device' argument
def load_ckpt(checkpoint_path, model, optimizer=None, scheduler=None, device=torch.device('cpu'), mode='eval'):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    if mode == 'eval':
        model.eval() 
        return model
    else:
        model.train()
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if scheduler is not None:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            return model, optimizer, scheduler
        else:
            return model, optimizer

# utility function to save a checkpoint (model_state, optimizer_state, scheduler_state) - saves on cpu (to save gpu memory)
def save_ckpt(device, checkpoint_path, model, optimizer, scheduler=None):
    # transfer model to cpu
    model = model.to('cpu')
    # prepare dicts for saving
    save_dict = {'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}
    if scheduler is not None:
        save_dict['scheduler_state_dict'] = scheduler.state_dict()
    torch.save(save_dict, checkpoint_path)
    # load model back on original device 
    model = model.to(device)


# main 
if __name__ == '__main__':
    latent_dim = 4 # embedding dimension of latent vectors
    img_size = 28 # mnist
    img_channels = 1
    img_shape = torch.tensor([img_channels, img_size, img_size])

    patch_size = 4 # 8 # so that we have 16 latents per img which should be sufficient
    assert img_size % patch_size == 0
    patch_dim = img_channels * (patch_size**2)

    seq_len = (img_size // patch_size) ** 2 # equal to num latents per item
    
    d_model = patch_dim * 1
    n_heads = 8
    assert d_model % n_heads == 0
    d_k = d_model // n_heads 
    d_v = d_k 
    n_layers = 6
    d_ff = d_model * 4
    dropout = 0.1

    max_epochs = 300
    epochs_done = 100
    batch_size = 512
    lr = 3e-4
    random_seed = 1010
    resume_training_from_ckpt = True          

    hyperparam_dict = {}
    hyperparam_dict['Ldim'] = latent_dim 
    hyperparam_dict['imgSize'] = img_size 
    hyperparam_dict['patchSize'] = patch_size 
    hyperparam_dict['patchDim'] = patch_dim 
    hyperparam_dict['seqLen'] = seq_len 
    hyperparam_dict['dModel'] = d_model 
    hyperparam_dict['nHeads'] = n_heads 
    hyperparam_dict['dropout'] = dropout     

    hyperparam_str = ''
    for k,v in hyperparam_dict.items():
        hyperparam_str += '|' + k + ':' + str(v) 

    save_folder = './plots/out_imgs_VAE_transformer_mnist' + hyperparam_str + '/'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    checkpoint_path = './ckpts/VAE_transformer_mnist' + hyperparam_str + '.pth' # path to a save and load checkpoint of the trained model       

    # set random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # cuda
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load CIFAR10 dataset
    resize_shape = (img_size, img_size)
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Resize(img_size, antialias=True ),  # image_size + 1/4 * image_size
        # torchvision.transforms.RandomResizedCrop(resize_shape, scale=(0.8, 1.0), antialias=True),
        # normalize img to appromately have mean=0 and std=1 (so imgs that were in range [0,1] will get zero centered but not strictly clamped in the range [-1,1])
        # torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # imagenet stats for normalization
        torchvision.transforms.Normalize(0.5, 0.5) 
    ])
    dataset = MNIST(root='./dataset_mnist', download=True, transform=transforms)
    dataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=16)

    # init transformer encoder
    encoder_transformer = init_transformer(patch_dim, seq_len, d_model, d_k, d_v, n_heads, n_layers, d_ff, dropout, latent_dim * 2, device)
    # init transformer decoder
    decoder_transformer = init_transformer(latent_dim, seq_len, d_model, d_k, d_v, n_heads, n_layers, d_ff, dropout, patch_dim, device)
    # init FSQ_Transformer 
    model = VAE_Transformer(latent_dim, encoder_transformer, decoder_transformer, seq_len, device).to(device)

    # init loss_fn and optimizer
    reconstruction_loss_function = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=lr) # no weight decay ?

    if resume_training_from_ckpt:
        model, optimizer = load_ckpt(checkpoint_path, model, optimizer, device=device, mode='train')

    losses = []

    # train 
    for epoch in tqdm(range(max_epochs)):
        epoch += epochs_done 
        train_loss = 0

        for batch_idx, data in enumerate(dataLoader):
            imgs, _ = data
            imgs = imgs.to(device)

            # convert img to sequence of patches
            x = img_to_patch_seq(imgs, patch_size, seq_len) # x.shape: [b, seq_len, patch_dim]

            recon_x, mu, logvar = model(x)

            loss = loss_function(recon_x, x, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.data
            optimizer.step()
            # if batch_idx % 50 == 0:
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch,
            #         batch_idx * len(imgs),
            #         len(dataLoader.dataset), 100. * batch_idx / len(dataLoader),
            #         loss.data / len(imgs)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(dataLoader.dataset)))
        losses.append(train_loss.item() / len(dataLoader.dataset))

        if epoch % (max_epochs // 40) == 0:
            # convert patch sequence to img 
            recon_imgs = patch_seq_to_img(recon_x, patch_size, img_channels)

            x = to_img(imgs.data)
            x_r = to_img(recon_imgs.data)
            # img_generated = generate_img(model, mu.shape[1:], device)
            save_image_recon(x[0], x_r[0], save_folder + '{}_reconstructed.png'.format(epoch))
            # save_image_gen(img_generated[0], save_folder + '{}_generated.png'.format(epoch))

    #save model
    save_ckpt(device, checkpoint_path, model, optimizer)

    # plot results
    plt.plot(losses, label='train_loss')
    plt.legend()
    plt.title('final_train_loss: ' + str(losses[-1]))
    plt.savefig(save_folder + 'plot_' + str(epoch) + '.png')
