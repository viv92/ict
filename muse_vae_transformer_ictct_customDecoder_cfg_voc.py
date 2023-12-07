'''
### Program implementing improved consistency training (iCT) using a transformer backbone 

## Features:
1. Consistency models are a class of generative models that generate data by learning to map a noised data point x_t to the denoised data point x_0.
2. Under the probability flow ODE framework, a data point x_0 is noised using a deterministic forward process represented by an ODE.
3. Since the process is deterministic, all noised versions of a data point (x_t for different t) can be uniquely mapped to the same starting data piont x_0.
4. So a function mapping x_t to x_0 satisfies the consistency property: f(x_t, t) = f(x_t', t') = x_0
5. Consistency models are related to score matching based generative models since the probability flow ODE governing the noising process is equivalent in distribution to a stochastic ODE governing a diffusion process.
6. Since the reverse process of a diffusion process is also a diffusion process involving the score term, the equivalent reverse probability flow ODE also involves the score term.
7. So learning the consistency function f(x_t, t) requires access to the score function. in consistency distillation method, we use a pretrained diffusion model to access the score information. While in consistency training, we use a particular form of probability flow ODE in which the score function can be empirically evaluated using monte-carlo estimate of another function: -(x_t - x) / t^2.   

## Todos / Questions:
1. [done] include schedules for steps N and EMA decay rate u
2. distance metric - using pseudo huber loss
3. this is a discrete time implementation. Try continuous time version later.
4. [done] modify consistency function to ensure boundary conditions for the consistency function
5. is the weighting factor in the loss: lambda_n = 1 always ? (Ans: No)
6. in algo 3, is there a difference between the step n (sampled uniformly) and corresponding time step t_n ?
7. resolve the confusion between train_step k and train_epoch (confusion between N, K and T)
8. note that x_T ~ gaussian(0, T^2) and not x_T ~ gaussian(0, identity)
9. note that forgoing resizing and random cropping in the dataloader transform gives a substantial speedup
10. note that the loss curve need not be decreasing (even when training is going well) as it gets arbitrarily weighted by lambdas

'''

import os
import cv2
import math 
from copy import deepcopy 
from matplotlib import pyplot as plt 
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import json 
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid

# import T5 (we use the T5 Encoder only)
from transformers import T5Tokenizer, T5ForConditionalGeneration

# import VAE for loading the pretrained weights
from VAE_transformer import VAE_Transformer, init_transformer, patch_seq_to_img, img_to_patch_seq

from utils_dpct_gendim_crossattn import *


# function implementing the consistency function - unet + boundary condition 
def consistency_function(net, x, t, condition, sigma_data, t_eps):
    c_in = 1 / torch.sqrt( torch.pow(t, 2) + sigma_data ** 2 )
    c_skip = (sigma_data ** 2) / ( torch.pow((t - t_eps), 2) + (sigma_data ** 2) )
    c_out = (sigma_data * (t - t_eps)) / torch.sqrt( (sigma_data ** 2) + torch.pow(t, 2) )
    # expand dims 
    c_skip = expand_dims_to_match(c_skip, x)
    c_out = expand_dims_to_match(c_out, x)
    c_in = expand_dims_to_match(c_in, x)
    if use_c_in:
        x_in = x * c_in
    else:
        x_in = x 
    out = net(x_in, t, condition)
    return c_skip * x + c_out * out 

# function to map discrete step interval [1, N] to continuous time interval [t_eps, T]
def step_to_time(rho, t_eps, T, N, n):
    inv_rho = 1/rho 
    a = math.pow(t_eps, inv_rho)
    b = math.pow(T, inv_rho)
    return torch.pow( a + ((b-a) * (n-1))/(N-1), rho) 

# function to calculate the list of time steps for a given schedule
def calculate_ts(rho, t_eps, T, N):
    # ts = [-1] # dummy element to offset indices starting from 1 and going upto N
    ts = [] # NOTE that in this implementation ts[0] corresponds to n=1
    for n in range(1, N+1):
        t_n = step_to_time(rho, t_eps, T, N, torch.tensor(n))
        ts.append(t_n)
    return torch.tensor(ts) 

# function to calculate lognormal distribution for steps (instead of uniform distribution)
def lognormal_step_distribution(ts, mean = -1.1, std = 2.0):
    pdf = torch.erf((torch.log(ts[1:]) - mean) / (std * math.sqrt(2))) - torch.erf(
        (torch.log(ts[:-1]) - mean) / (std * math.sqrt(2))
    )
    pdf = pdf / pdf.sum()
    return pdf 

# function to calculate all loss weight factors (lambdas) 
def calculate_lambdas(ts):
    return 1 / (ts[1:] - ts[:-1]) 

# function imlementing dicrete step schedule - returns the discrete step value N_k for a training step k  
def discrete_step_schedule_improved(s0, s1, K, k):
    a = (s1+1-s0) / 2
    b = 1 - math.cos( (math.pi * k) / K)
    return  math.ceil((a * b) + s0 - 1) + 1

# function implementing pseudo huber loss 
def pseudo_huber_loss(x,y):
    c = 0.00054 * math.sqrt( math.prod(x.shape[1:]) )
    return torch.sqrt( torch.pow(x-y,2) + c**2 ) - c 

# function to expand dims of tensor x to match that of tensor y 
def expand_dims_to_match(x, y):
    while len(x.shape) < len(y.shape):
        x = x.unsqueeze(-1)
    return x 


# function to sample / generate img 
def sample(net, img_shape, rho, start_time, end_time, class_label, sigma_data, sampling_steps):
    z = torch.randn(img_shape) * end_time # NOTE that initial noise x_T ~ gaussian(0, T^2) and not x_T ~ gaussian(0, identity)
    T = torch.tensor(end_time).unsqueeze(0).expand(img_shape[0]) # expand to n_samples
    x = consistency_function(net, z.to(device), T.to(device), class_label.to(device), sigma_data, start_time)
    # note that sampling_steps = N-1 in algo 1
    for n in range(sampling_steps, 0, -1): 
        z = torch.randn(img_shape)
        t = step_to_time(rho, start_time, end_time, sampling_steps+1, torch.tensor(n).long())
        t = t.unsqueeze(0).expand(img_shape[0]) # expand to n_samples
        # if n > 1: # since a = 0 for n = 1
        a = torch.sqrt( torch.pow(t,2) - (start_time**2) )
        x = x + expand_dims_to_match(a, x).to(device) * z.to(device)
        x = consistency_function(net, x, t.to(device), class_label.to(device), sigma_data, start_time) 
    return x 


# function to sample / generate img - but the sampling time sequence is linear
def sample_linear_cfg(net, img_shape, rho, start_time, end_time, class_label, sigma_data, sampling_steps, cfg_scale):
    ts = [start_time + i * ((end_time - start_time) / sampling_steps) for i in range(sampling_steps+1)] 
    ts = ts[::-1] # sampling time sequence
    z = torch.randn(img_shape) * end_time # NOTE that initial noise x_T ~ gaussian(0, T^2) and not x_T ~ gaussian(0, identity)
    T = torch.tensor(ts[0]).unsqueeze(0).expand(img_shape[0]) # expand to n_samples
    x_cond = consistency_function(net, z.to(device), T.to(device), class_label.to(device), sigma_data, start_time)
    x_uncond = consistency_function(net, z.to(device), T.to(device), None, sigma_data, start_time)
    # note that sampling_steps = N in algo 1
    for n in range(1, sampling_steps+1): 
        z = torch.randn(img_shape)
        t = torch.tensor(ts[n]).unsqueeze(0).expand(img_shape[0]) # expand to n_samples
        # if n < sampling_steps-1 : # since a = 0 for n = 1
        a = torch.sqrt( torch.pow(t,2) - (start_time**2) )
        z_cond = x_cond + expand_dims_to_match(a, x_cond).to(device) * z.to(device)
        z_uncond = x_uncond + expand_dims_to_match(a, x_uncond).to(device) * z.to(device)
        # interpolate z_cond and z_uncond 
        zn = z_cond + cfg_scale * (z_cond - z_uncond)
        # predict / denoise 
        x_cond = consistency_function(net, zn, t.to(device), class_label.to(device), sigma_data, start_time) 
        x_uncond = consistency_function(net, zn, t.to(device), None, sigma_data, start_time) 
    return x_cond 


# utility function to load img and captions data 
def load_data():
    imgs_folder = '/home/vivswan/experiments/muse/dataset_coco_val2017/images/'
    captions_file_path = '/home/vivswan/experiments/muse/dataset_coco_val2017/annotations/captions_val2017.json'
    captions_file = open(captions_file_path)
    captions = json.load(captions_file)
    img_dict, img_cap_pairs = {}, []
    print('Loading Data...')
    num_iters = len(captions['images']) + len(captions['annotations'])
    pbar = tqdm(total=num_iters)

    for img in captions['images']:
        id, file_name = img['id'], img['file_name']
        img_dict[id] = file_name
    for cap in captions['annotations']:
        id, caption = cap['image_id'], cap['caption']
        # use img_name as key for img_cap_dict
        img_filename = img_dict[id]

        # load image from img path 
        img_path = imgs_folder + img_filename
        img = cv2.imread(img_path, 1)
        resize_shape = (img_size, img_size)
        img = cv2.resize(img, resize_shape, interpolation=cv2.INTER_LINEAR)
        img = np.float32(img) / 255
        img = torch.tensor(img)
        img = img.permute(2, 0, 1) # [w,h,c] -> [c,h,w]
        transforms = torchvision.transforms.Compose([
            # NOTE: no random resizing as its asking the model to learn to predict different img tokens for the same caption
            # torchvision.transforms.Resize( int(1.25*img_size) ),  # image_size + 1/4 * image_size
            # torchvision.transforms.RandomResizedCrop(resize_shape, scale=(0.8, 1.0)),
            # torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # imagenet stats for normalization
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # zero mean, unit std
        ])
        img = transforms(img)

        # img_cap_pairs.append([img_filename, caption])
        img_cap_pairs.append([img, caption])
        pbar.update(1)
    pbar.close()
    return img_cap_pairs


# utility function to process minibatch - convert img_filenames to augmented_imgs and convert caption_text to tokenized_captions - then obtain embeddings for them
def process_batch(minibatch, tokenizer, img_size, device):
    augmented_imgs, captions = list(map(list, zip(*minibatch)))

    # augmented_imgs = []
    # img_files, captions = list(map(list, zip(*minibatch)))

    # tokenize captions 
    caption_tokens_dict = tokenizer(captions, return_tensors='pt', padding=True, truncation=True)

    # # get augmented imgs
    # imgs_folder = '/home/vivswan/experiments/muse/dataset_coco_val2017/images/'
    # for img_filename in img_files:
    #     img_path = imgs_folder + img_filename
    #     img = cv2.imread(img_path, 1)
    #     resize_shape = (img_size, img_size)
    #     img = cv2.resize(img, resize_shape, interpolation=cv2.INTER_LINEAR)
    #     img = np.float32(img) / 255
    #     img = torch.tensor(img)
    #     img = img.permute(2, 0, 1) # [w,h,c] -> [c,h,w]
    #     transforms = torchvision.transforms.Compose([
    #         # NOTE: no random resizing as its asking the model to learn to predict different img tokens for the same caption
    #         # torchvision.transforms.Resize( int(1.25*img_size) ),  # image_size + 1/4 * image_size
    #         # torchvision.transforms.RandomResizedCrop(resize_shape, scale=(0.8, 1.0)),
    #         # torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # imagenet stats for normalization
    #         torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # zero mean, unit std
    #     ])
    #     img = transforms(img)
    #     augmented_imgs.append(img)

    augmented_imgs = torch.stack(augmented_imgs, dim=0).to(device)
    caption_tokens_dict = caption_tokens_dict.to(device)
    return augmented_imgs, caption_tokens_dict


# utility function to freeze model
def freeze(model):
    for p in model.parameters():
        p.requires_grad_(False) 

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
        

# convert tensor to img
def to_img(x):
    x = 0.5 * x + 0.5 # transform img from range [-1, 1] -> [0, 1]
    x = x.clamp(0, 1) # clamp img to be strictly in [-1, 1]
    x = x.permute(0,2,3,1) # [b,c,h,w] -> [b,h,w,c]
    return x 

# function to save a generated img
def save_img_generated(x_g, save_path):
    gen_img = x_g.detach().cpu().numpy()
    gen_img = np.uint8( gen_img * 255 )
    # bgr to rgb 
    # gen_img = gen_img[:, :, ::-1]
    cv2.imwrite(save_path, gen_img)
        


### main
if __name__ == '__main__':
    # hyperparams for vqvae (FSQ_Transformer)
    latent_dim = 16
    img_size = 128 # voc
    img_channels = 3
    img_shape = torch.tensor([img_channels, img_size, img_size])
    resize_shape = (img_size, img_size)
    img_latent_dim = latent_dim # as used in the pretrained VQVAE 

    patch_size = 16 # # necessary that img_size % patch_size == 0
    assert img_size % patch_size == 0
    patch_dim = img_channels * (patch_size**2)
    seq_len = (img_size // patch_size) ** 2 # equal to num latents per item
    
    # hyperparams for VAE Transformer
    d_model_vae = patch_dim * 1
    n_heads_vae = 8
    assert d_model_vae % n_heads_vae == 0
    d_k_vae = d_model_vae // n_heads_vae 
    d_v_vae = d_k_vae 
    n_layers_vae = 6
    d_ff_vae = d_model_vae * 4
    dropout_vae = 0.1

    # hyperparams for custom decoder (DPCT)
    d_model_dpct = latent_dim * 64 # 32 
    n_heads_dpct = 2 # 8
    assert d_model_dpct % n_heads_dpct == 0
    d_k_dpct = d_model_dpct // n_heads_dpct 
    d_v_dpct = d_k_dpct 
    n_layers_dpct = 6
    d_ff_dpct = d_model_dpct * 4
    dropout_dpct = 0.

    # hyperparams for T5 (T5 decoder implements the consistency model backbone)
    d_model_t5 = 768 # d_model for T5 (required for image latents projection)
    max_seq_len_t5 = 512 # required to init T5 Tokenizer
    # dropout = 0. # TODO: check if we can set the dropout in T5 decoder

    # hyperparams for consistency training
    start_time = 0.002 # start time t_eps of the ODE - the time interval is [t_eps, T] (continuous) and corresponding step interval is [1, N] (discrete)
    end_time = 80 # 16 # end time T of the ODE (decreasing end time leads to lower loss with some improvement in sample quality)
    N_initial = 10
    N_final = 1023 # final value of N in the step schedule (denoted as s_1 in appendix C)
    rho = 7.0 # used to calculate mapping from discrete step interval [1, N] to continuous time interval [t_eps, T]
    sigma_data = 0.5 # used to calculate c_skip and c_out to ensure boundary condition
    P_mean = -1.1 # mean of the train time noise sampling distribution (log-normal)
    P_std = 2.0 # std of the train time noise sampling distribution (log-normal)
    use_c_in = True  

    sampling_start_time = math.exp( (math.log(start_time) + math.log(end_time)) / 2)
    sampling_strategy = 'linearCFGCorrected'
    sampling_net = 'ema'
    n_samples = 4 

    num_diffusion_processes = int( (N_final - N_initial) * 1.0 ) # 0.6 # smaller jumps between N_k values = higher sample quality (no effect on loss though)
    num_train_steps_per_diffusion_process = 47
    total_train_steps = num_diffusion_processes * num_train_steps_per_diffusion_process
    ema_decay_rate = 0.999 # fixed decay rate for student ema network 

    lr = 1e-4 # 3e-4 
    batch_size = 256 # lower batch size allows for more training steps per diffusion process (but reduces compute efficiency)
    random_seed = 10
    sample_freq = int(total_train_steps / 100)
    model_save_freq = int(total_train_steps / 10)
    plot_freq = model_save_freq
    p_uncond = 0.1 # for cfg
    cfg_scale = 1.5
    resume_training_from_ckpt = False               

    hyperparam_dict = {}
    # hyperparam_dict['t0'] = start_time
    # hyperparam_dict['tN'] = end_time
    hyperparam_dict['N_initial'] = N_initial
    hyperparam_dict['N_final'] = N_final
    # hyperparam_dict['sampleStartTime'] = '{:.2f}'.format(sampling_start_time)
    # hyperparam_dict['sampleStrategy'] = sampling_strategy
    hyperparam_dict['sampleNet'] = sampling_net
    hyperparam_dict['DP'] = num_diffusion_processes
    hyperparam_dict['steps1DP'] = num_train_steps_per_diffusion_process
    # hyperparam_dict['lr'] = lr
    # hyperparam_dict['batch'] = batch_size
    hyperparam_dict['dmodelDPCT'] = d_model_dpct
    hyperparam_dict['nheadsDPCT'] = n_heads_dpct
    hyperparam_dict['nlayersDPCT'] = n_layers_dpct
    hyperparam_dict['dropoutDPCT'] = dropout_dpct
    hyperparam_dict['pUncond'] = p_uncond
    hyperparam_dict['cfgScale'] = cfg_scale
    hyperparam_dict['useCin'] = use_c_in 

    hyperparam_str = ''
    for k,v in hyperparam_dict.items():
        hyperparam_str += '|' + k + ':' + str(v) 

    save_folder = './generated_muse_vae_transformer_ictct_customDecoder_cfg_voc' + hyperparam_str
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # vae ckpt path 
    vae_ckpt_path = '/home/vivswan/experiments/latent_diffusion/ckpts/VAE_transformer_voc|Ldim:16|imgSize:128|patchSize:16|patchDim:768|seqLen:64|dModel:768|nHeads:8|dropout:0.1.pth' # path to pretrained vqvae 

    # t5 model (for encoding captions) 
    t5_model_name = 't5-base'

    # muse_vqvae_edm save ckpt path
    muse_ckpt_path = './ckpts/muse_vae_transformer_ictct_customDecoder_cfg_voc' + hyperparam_str + '.pt'

    # set random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # create dataset from img_cap_dict
    dataset = load_data()
    dataset_len = len(dataset)

    # load pretrained VAE in eval mode 
    # init transformer encoder
    encoder_transformer = init_transformer(patch_dim, seq_len, d_model_vae, d_k_vae, d_v_vae, n_heads_vae, n_layers_vae, d_ff_vae, dropout_vae, latent_dim * 2, device)
    # init transformer decoder
    decoder_transformer = init_transformer(latent_dim, seq_len, d_model_vae, d_k_vae, d_v_vae, n_heads_vae, n_layers_vae, d_ff_vae, dropout_vae, patch_dim, device)
    # init VAE_Transformer 
    vae_model = VAE_Transformer(latent_dim, encoder_transformer, decoder_transformer, seq_len, device).to(device)
    vae_model = load_ckpt(vae_ckpt_path, vae_model, device=device, mode='eval')

    # init T5 tokenizer and transformer model
    t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name, model_max_length=max_seq_len_t5)
    t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name).to(device)

    # delete t5_decoder to save ram 
    del t5_model.decoder 

    # init custom decoder (DPCT)
    max_seq_len_dpct = seq_len * 2 + 1 # [t, x_noised, x_denoised]
    condition_dim = d_model_t5 
    net = init_dpct(max_seq_len_dpct, seq_len, d_model_dpct, latent_dim, condition_dim, d_k_dpct, d_v_dpct, n_heads_dpct, n_layers_dpct, d_ff_dpct, dropout_dpct, device).to(device)

    # create target network copy 
    ema_net = deepcopy(net)

    # freeze vqvae, t5_encoder and ema_net
    freeze(vae_model)
    freeze(t5_model.encoder)
    freeze(ema_net)

    # optimizer and loss criterion
    optimizer = torch.optim.AdamW(params=net.parameters(), lr=lr)

    # load ckpt
    if resume_training_from_ckpt:
        net, optimizer = load_ckpt(muse_ckpt_path, net, optimizer, device=device, mode='train')

    # train

    prev_k = -1 # to keep track of when to update schedule
    N_k = N_initial 
    u_0 = ema_decay_rate

    train_step = 0
    epoch = 0
    ema_losses = []
    criterion = nn.MSELoss(reduction='none') # NOTE that reduction=None is necessary so that we can apply weighing factor lambda

    ts = calculate_ts(rho, start_time, end_time, N_final) # NOTE this is used only for sampling in the EDM approach

    pbar = tqdm(total=total_train_steps)
    while train_step < total_train_steps:

        # fetch minibatch
        idx = np.arange(dataset_len)
        np.random.shuffle(idx)
        idx = idx[:batch_size]
        minibatch = [dataset[i] for i in idx]

        # process minibatch - convert img_filenames to augmented_imgs and convert caption_text to tokenized_captions
        # note that we don't create embeddings yet, since that's done by the image_encoder and T5 model
        imgs, cap_tokens_dict = process_batch(minibatch, t5_tokenizer, img_size, device) # imgs.shape:[batch_size, 3, 32, 32], captions.shape:[batch_size, max_seq_len]

        with torch.no_grad():
            # convert img to sequence of patches
            x = img_to_patch_seq(imgs, patch_size, seq_len) # x.shape: [b, seq_len, patch_dim]

            mu, logvar = vae_model.encode(x)
            img_latents = vae_model.reparameterize(mu, logvar)

            # extract cap tokens and attn_mask from cap_tokens_dict
            cap_tokens, cap_attn_mask = cap_tokens_dict.input_ids, cap_tokens_dict.attention_mask
            # feed cap_tokens to t5 encoder to get encoder output
            enc_out = t5_model.encoder(input_ids=cap_tokens, attention_mask=cap_attn_mask).last_hidden_state # enc_out.shape: [batch_size, cap_seqlen, d_model_t5]

        x = img_latents 
        y = enc_out 

        # for sampling 
        sample_caption_emb = y[:1] # NOTE that we sample one class label but generate n_sample imgs for that label
        sample_caption_emb = sample_caption_emb.expand(n_samples, -1, -1)

        # set labels = None with prob p_uncond
        if np.random.rand() < p_uncond: # TODO: explore the effect of no CFG versus CFG only during training versus CFG during training and sampling
            y = None


        # get new step schedule if k changed (increased)
        k = int(train_step / num_train_steps_per_diffusion_process) 
        # k = k + 1
        if not (k == prev_k):
            prev_k = k 
            N_k = discrete_step_schedule_improved( N_initial, N_final, num_diffusion_processes, k )
            # u_k = ema_decay_rate_schedule(N_initial, ema_decay_rate, N_k) # current decay rate
            ts = calculate_ts(rho, start_time, end_time, N_k) # NOTE that in this implementation ts[0] corresponds to n=1
            step_pdf = lognormal_step_distribution(ts, P_mean, P_std)
            lambdas = calculate_lambdas(ts) 

        # sample step using log normal distribution
        n = torch.multinomial(step_pdf, x.shape[0], replacement=True)

        # get corresponding time steps
        t_n = ts[n].to(device)
        t_n_plus1 = ts[n+1].to(device)
        # get corresponding noised data points
        z = torch.randn_like(x)
        x_n = x + expand_dims_to_match(t_n, x) * z 
        x_n_plus1 = x + expand_dims_to_match(t_n_plus1, x) * z
        # predict x_0 
        pred_n_plus1 = consistency_function(net, x_n_plus1, t_n_plus1, y, sigma_data, start_time)
        # predict target x_0 
        with torch.no_grad():
            # NOTE that we use the student net and not ema net
            # ema net is used only for sampling 
            pred_n = consistency_function(net, x_n, t_n, y, sigma_data, start_time) 
        
        # distance metric is pseudo huber loss 
        d = pseudo_huber_loss(pred_n_plus1, pred_n.detach())
        weight_factor = expand_dims_to_match(lambdas[n], d).to(device)
        loss = weight_factor * d
        loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update ema_losses (for plotting)
        if ema_losses == []:
            prev_loss = loss.item()
        else:
            prev_loss = ema_losses[-1]
        curr_loss = prev_loss * 0.9 + loss.item() * 0.1
        ema_losses.append(curr_loss)

        # update ema_net weights using ema decay
        with torch.no_grad():
            for target_param, current_param in zip(ema_net.parameters(), net.parameters()):
                # target_param.data.copy_( u_k * target_param.data + (1-u_k) * current_param.data )
                target_param.mul_(u_0).add_(current_param, alpha=1-u_0)

        train_step += 1
        pbar.update(1)
        pbar.set_description('N_k:{} loss:{:.10f}'.format(N_k, ema_losses[-1]))

        # save ckpt 
        if train_step % model_save_freq == 0:
            save_ckpt(device, muse_ckpt_path, net, optimizer)

        # sample
        if train_step % sample_freq == 0:
            
            net.eval()

            # sample points - equivalent to just evaluating the consistency function
            with torch.no_grad():

                if sampling_net == 'ema':
                    samp_net = ema_net
                else:
                    samp_net = net 

                # take a single caption and repeat it n_samples time
                sample_caption_string = minibatch[0][1]
                # handle illegal captions
                sample_caption_string = sample_caption_string[:50]
                if '/' in sample_caption_string:
                    sample_caption_string = 'null'

                sample_shape = x[:n_samples].shape # since we want to sample 'n_sample' points

                sampling_steps = [3]
                for samp_step in sampling_steps:
                    # sample_caption_emb = sample_caption_emb.unsqueeze(-1).unsqueeze(-1).float()
                    if sampling_strategy == 'log':
                        sampled_img_latents = sample(samp_net, sample_shape, rho, sampling_start_time, end_time, sample_caption_emb, sigma_data, samp_step)
                    else:
                        sampled_img_latents = sample_linear_cfg(samp_net, sample_shape, rho, sampling_start_time, end_time, sample_caption_emb, sigma_data, samp_step-1, cfg_scale)


                # decode img latents to pixels using vae decoder
                gen_img_patch_seq = vae_model.decode(sampled_img_latents)

                # convert patch sequence to img 
                gen_imgs = patch_seq_to_img(gen_img_patch_seq, patch_size, img_channels) # [b,c,h,w]

                ori_img = imgs[0] # [c,h,w]
                ori_img = ori_img.permute(1,2,0) # [h,w,c]
                ori_img = (ori_img * 0.5 + 0.5).clamp(0,1)

                # save ori img
                save_img_name = 'trainStep=' + str(train_step) + '_caption=' + sample_caption_string + '_original.png'
                save_path = save_folder + '/' + save_img_name
                save_img_generated(ori_img, save_path)

                gen_imgs = (gen_imgs * 0.5 + 0.5).clamp(0,1) # [b,c,h,w]
                # bgr to rgb 
                gen_imgs = torch.flip(gen_imgs, dims=(1,))
                grid = make_grid(gen_imgs, nrow=2)
                save_image(grid, f"{save_folder}/trainStep={train_step}_caption={sample_caption_string}.png")

            net.train()

        if train_step % plot_freq == 0:
            # plot losses 
            plt.figure()
            l = int( len(ema_losses) / 2 ) # for clipping first half
            plt.plot(ema_losses[l:])
            plt.title('final_loss:{:.10f}'.format(ema_losses[-1]))
            plt.savefig(save_folder + f'/loss_trainStep={train_step}.png' )

    epoch += 1
    pbar.update(1)


    pbar.close()
