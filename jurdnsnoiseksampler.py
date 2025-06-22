import torch
import comfy.samplers
import comfy.sample
import comfy.model_management

class KSamplerWithIterativeNoise:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "add_noise_every_n_steps": ("INT", {"default": 1, "min": 1, "max": 100}),
                "start_noise_strength": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 2.0, "step": 0.01}),
                "end_noise_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling/custom"

    def sample(self, model, positive, negative, latent_image, seed, steps, cfg, sampler_name, scheduler, add_noise_every_n_steps, start_noise_strength, end_noise_strength, denoise=1.0):
        device = comfy.model_management.get_torch_device()
        latent_img = latent_image["samples"]
        latent_img = comfy.sample.fix_empty_latent_channels(model, latent_img)
        batch_inds = latent_image.get("batch_index", None)
        noise = comfy.sample.prepare_noise(latent_img, seed, batch_inds)
        noise_mask = latent_image.get("noise_mask", None)
        sampler = comfy.samplers.ksampler(sampler_name)
        sigmas = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), scheduler, steps).to(device)
        
        def iterative_noise_callback(step, x0, x, total_steps):
            if step > 0 and (step % add_noise_every_n_steps == 0):
                progress = step / (total_steps - 1) if total_steps > 1 else 1.0
                current_noise_strength = start_noise_strength + (end_noise_strength - start_noise_strength) * progress
                if current_noise_strength > 0:
                    current_sigma = sigmas[step] if step < len(sigmas) else sigmas[-1]
                    perturb_generator = torch.Generator(device=device).manual_seed(seed + step + 1)
                    perturb_noise = torch.randn(x.shape, generator=perturb_generator, device=device, dtype=x.dtype)
                    x += perturb_noise * current_sigma * current_noise_strength
        
        samples = comfy.sample.sample(model=model, noise=noise, steps=steps, cfg=cfg, sampler_name=sampler_name, scheduler=scheduler, positive=positive, negative=negative, latent_image=latent_img, denoise=denoise, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False, noise_mask=noise_mask, callback=iterative_noise_callback, disable_pbar=False, seed=seed)
        return ({"samples": samples},)

NODE_CLASS_MAPPINGS = {"KSamplerIterativeNoise": KSamplerWithIterativeNoise}
NODE_DISPLAY_NAME_MAPPINGS = {"KSamplerIterativeNoise": "Jurdns Iterative Noise KSampler"}