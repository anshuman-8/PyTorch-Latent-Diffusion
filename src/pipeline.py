import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torch.nn.functional as F
from tqdm import tqdm
from ddpm import DDPMSampler

HEIGHT = 512
WIDTH = 512
LATENT_DIM = 512 // 8  # both height and width are halved 3 times
MAX_TOKENS = 77


def generate(
    prompt: str,
    negetive_prompt: str,
    input_img=None,
    strength=0.8,
    is_cfg: bool = True,
    cfg_scale: int = 7.5,
    sampler_name="ddpm",
    inference_steps=50,
    model={},
    seed=None,
    device=None,
    idle_device = None,
    tokenizer=None,
):
    with torch.no_grad():
        if not (0<=strength<=1):
            raise ValueError("strength must be between 0 and 1")
        if not (0<=cfg_scale<=17.5):
            raise ValueError("cfg_scale must be between 0 and 10")
        if not (0<=inference_steps<=200):
            raise ValueError("inference_steps must be between 0 and 100")
        
        if idle_device:
            to_idle_device = lambda x: x.to(idle_device)
        else:
            to_idle_device = lambda x: x

        generator = torch.Generator(device=device)
        if seed is not None:
            generator.manual_seed(seed)
        else:
            generator.manual_seed(torch.seed())

        clip = model['clip']
        clip.to(device)
        
        if is_cfg:
            cond_token = tokenizer.batch_encode_plus([prompt], padding="max_length", max_lenght=MAX_TOKENS).input_ids # tokenize the prompt
            cond_token = torch.tensor(cond_token, dtype=torch.long, device=device)
            cond_token = clip(cond_token)

            uncond_token = tokenizer.batch_encode_plus([negetive_prompt], padding="max_length", max_lenght=MAX_TOKENS).input_ids # tokenize the prompt
            uncond_token = torch.tensor(uncond_token, dtype=torch.long, device=device)
            uncond_token = clip(uncond_token)

            context = torch.cat([cond_token, uncond_token], dim=0)

        else:
            token = tokenizer.batch_encode_plus([prompt], padding="max_length", max_lenght=MAX_TOKENS).input_ids
            token = torch.tensor(token, dtype=torch.long, device=device)

            context = clip(token)

        to_idle_device(clip) # move the model to the idle device, offload model after using it

        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference
        # elif sampler_name == "ddim": 
        #     sampler = DDIMSampler(generator)
        else:
            raise ValueError("sampler_name must be either 'ddpm' or 'ddim'")
        
        latent_shape = (1, 4, LATENT_DIM, LATENT_DIM)

        if input_img:
            encoder = model['encoder']
            encoder.to(device)

            input_image_numpy = input_img.resize((HEIGHT, WIDTH), Image.LANCZOS)
            input_image_numpy = np.array(input_image_numpy)
            input_image_tensor = torch.tensor(input_image_numpy, dtype=torch.float, device=device)
            input_image_tensor = rescale(input_image_tensor, (0,255), (-1, 1)) # rescale to -1 to 1 from 0 to 255
            input_image_tensor = input_image_tensor.unsqueeze(0)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

            encoder_noise = torch.randn(latent_shape, device=device, generator=generator)

            latents = encoder(input_image_tensor, encoder_noise)

            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timestep[0])

            to_idle_device(encoder) # move the model to the idle device, offload model after using it

        else:
            # random noise for text to image
            latents = torch.randn(latent_shape, device=device, generator=generator)

        diffusion = model['diffusion']
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps)

        for i, timesteps in enumerate(timesteps):
            timesteps = get_time_embedding(timesteps).to(device) # equivalent to positional embedding of transformer

            model_input = latents # (b, 4, 64, 64)

            if is_cfg:
                model_input = model_input.repeat(2, 1, 1, 1) # (b*2, 4, 64, 64), one cond and one uncond

            # predicted noise by the model
            model_output = diffusion(model_input, context, timesteps) # (b*2, 4, 64, 64)

            if is_cfg:
                # model_output = model_output[:1]
                opt_cond, opt_uncond  = model_output.chunk(2, dim=0)

                model_output = cfg_scale * (opt_cond - opt_uncond) + opt_uncond # (b, 4, 64, 64)

            # remove noise predicted by the unet
            latents = sampler.step(latents, model_output, timesteps)

        to_idle_device(diffusion) 

        decoder = model['decoder']
        decoder.to(device) 

        image = decoder(latents)     

        to_idle_device(decoder)

        image = rescale(image, (-1, 1), (0, 255)) # rescale to 0 to 255 from -1 to 1
        image = image.squeeze(0).permute(1, 2, 0) # (3, 512, 512) -> (512, 512, 3)
                
        image = image.to('cpu').numpy().astype(np.uint8)

        return Image.fromarray(image)
    
def rescale(image, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range

    image = image - old_min
    image = image / (old_max - old_min)

    image = image * (new_max - new_min)
    image = image + new_min

    if clamp:
        image = image.clamp(min=new_min, max=new_max)

    return image


def get_time_embedding(timesteps: torch.Tensor) -> torch.Tensor:
    freqs = torch.pow(10000, -torch.arange(start=0, end=100, dtype=torch.float32, device=timesteps.device) / 160)
    x =  torch.tensor([timesteps], dtype=torch.float32)[:, None] * freqs[None, :]
    emmb = torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
    return emmb



