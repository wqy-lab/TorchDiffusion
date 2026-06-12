import torch
import torch.nn

class Diffusion(object):
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, device='cpu'):
        self.T = T
        self.beta = torch.linspace(beta_start, beta_end, T, device=device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
    
    def q_sample(self, x0, t, noise):
        co1 = torch.sqrt(self.alpha_bar[t]).reshape(-1, 1, 1, 1)
        co2 = torch.sqrt(1 - self.alpha_bar[t]).reshape(-1, 1, 1, 1)
        return co1 * x0 + co2 * noise, noise

    @torch.no_grad()
    def p_sample(self, xt, t, pred_noise):
        alpha_t = self.alpha[t].reshape(1, 1, 1, 1)
        alpha_bar_t = self.alpha_bar[t].reshape(1, 1, 1, 1)
        beta_t = self.beta[t].reshape(1, 1, 1, 1)
        if t > 0:
            alpha_bar_prev = self.alpha_bar[t - 1].reshape(1, 1, 1, 1)
        else:
            alpha_bar_prev = torch.ones(1, 1, 1, 1, device=xt.device)

        mean = (1.0 / torch.sqrt(alpha_t)) * (
            xt - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * pred_noise
        )
        if t == 0:
            return mean

        variance = (1 - alpha_bar_prev) / (1 - alpha_bar_t) * beta_t
        return mean + torch.sqrt(variance) * torch.randn_like(xt)

    @torch.no_grad()
    def sample(self, model, batch_size, image_size, channels):
        device = self.beta.device
        x = torch.randn(batch_size, channels, image_size, image_size, device=device)
        for t in reversed(range(self.T)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            pred_noise = model(x, t_batch)
            x = self.p_sample(x, t, pred_noise)
        return x.clamp(0.0, 1.0)


