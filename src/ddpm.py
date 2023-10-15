import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

class DDPMSampler:
    def __init__(self, generator:torch.Generator, training_steps=1000, beta_start=0.00085, beta_end=0.0120):
        #   here beta start/end is the variance schedule, it gives the variance of the noice we add at each step
        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, training_steps, dtype=torch.float32)
        self.alpha = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0) # [alpha0, alpha0*alpha1, alpha0*alpha1*alpha2, ...]

        self.generator = generator
        self.training_steps = training_steps
        self.timesteps = torch.from_numpy(np.arange(0, training_steps)[::-1].copy())

        self.one = torch.tensor(1.0, dtype=torch.float32)
        
    def set_inference_timesteps(self, inference_steps:int=50):
        self.inference_steps = inference_steps
        step_ratio = inference_steps // self.training_steps
        timesteps = (np.arange(0, inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps)

    def add_noise(self, original_sample:torch.FloatTensor, timesteps:torch.IntTensor) -> torch.FloatTensor:
        alpha_cumprod = self.alpha_cumprod.to(device=original_sample.device, dtype=original_sample.dtype)
        timesteps = timesteps.to(device=original_sample.device)

        sqrt_alpha_cumprod = alpha_cumprod[timesteps] ** 0.5
        sqrt_alpha_cumprod = sqrt_alpha_cumprod.flatten()
        while len(sqrt_alpha_cumprod.shape) < len(original_sample.shape):
            sqrt_alpha_cumprod = sqrt_alpha_cumprod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alpha_cumprod[timesteps]) ** 0.5 # standard deviation
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_sample.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noise = torch.randn(original_sample.shape, generator=self.generator, device=original_sample.device, dtype=original_sample.dtype)

        noisy_sample = (sqrt_alpha_cumprod * original_sample) + (sqrt_one_minus_alpha_prod) * noise # X = mean + std * z

        return noisy_sample
    
    def set_strength(self, strength:int = 1):
        start_step = self.inference_steps - (self.inference_steps * strength)
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step
    
    def _get_prev_timestep(self, timestep:int) -> int:
        prev_timestep = timestep - (self.training_steps // self.inference_steps)
        return prev_timestep
    
    def _get_variance(self, timestep:int) -> torch.Tensor:
        prev_t = self._get_prev_timestep(timestep)

        alpha_t = self.alpha_cumprod[timestep]
        alpha_t_prev = self.alpha_cumprod[prev_t] if prev_t >= 0 else self.one
        current_beta_t = alpha_t / alpha_t_prev

        variance = (1 - alpha_t_prev) / (1 - alpha_t) * current_beta_t
        variance = torch.clamp(variance, min=1e-20)

        return variance

    
    def step(self, timestep:int, latents:torch.Tensor, model_output:torch.Tensor):
        t = timestep
        prev_t = self._get_prev_timestep(t)
        alpha_t = self.alpha_cumprod[t]
        alpha_t_prev = self.alpha_cumprod[prev_t] if prev_t >=0 else self.one
        beta_t = 1 - alpha_t
        beta_t_prev = 1 - alpha_t_prev
        current_alpha_t = alpha_t / alpha_t_prev
        current_beta_t = 1 - current_alpha_t

        # compute x0 i.e. predicted original sample using formula 15 in ddpm paper
        pred_original_sample = (latents - beta_t ** 0.5 * model_output) / alpha_t ** 0.5

        # compute the coefficient of pre_original_sample and current sample [eq 7 in ddpm paper]
        pred_original_sample_coeff = (alpha_t_prev ** 0.5 * beta_t) / (beta_t) 
        current_sample_coeff = (current_alpha_t ** 0.5 * beta_t_prev) / beta_t

        # compute the predicted previous sample mean 
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * latents

        variance = 0
        if t > 0:
            device = model_output.device
            noise = torch.randn(model_output.shape, generator=self.generator, device=device, dtype=model_output.dtype)
            variance = (self._get_variance(t) ** 0.5) * noise

        # N(0, 1) --> N(mu, sigma^2)
        # X = mu + sigma * Z where Z ~ N(0, 1)
        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample






