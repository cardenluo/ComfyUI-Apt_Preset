import nodes
import torch
import node_helpers



class sampler_SeedVariance:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "randomize_percent": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 100.0, "step": 1,}),  
                "strength": ("FLOAT", {"default": 20, "min": 0.0, "max": 1000000, "step": 0.01, }),
                "noise_insert": (["noise on beginning steps", "noise on ending steps", "noise on all steps"],),
                "steps_switchover_percent": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 100.0, "step": 1,}),
                "seed": ("INT", {"default": 1, "min": 0, "max": 0xFFFFFFFF, "step": 1})
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "randomize_conditioning"
    CATEGORY = "Apt_Preset/chx_ksample/😺backup"

    def randomize_conditioning(self, conditioning, strength, seed, randomize_percent, noise_insert, steps_switchover_percent):
        if randomize_percent == 0 or strength == 0:
            return (conditioning,)

        randomize_percent = randomize_percent / 100
        steps_switchover_percent = steps_switchover_percent / 100

        torch.manual_seed(seed)

        noisy_embedding = []
        for t in conditioning:
            if isinstance(t[0], torch.Tensor):
                noise = torch.rand_like(t[0]) * 2 * strength - strength
                mask = torch.bernoulli(torch.ones_like(t[0]) * randomize_percent).bool() 
                modified_noise = noise * mask  
                noisy_embedding.append([t[0] + modified_noise, t[1]])
            else:
                return (conditioning,)
            break # we will only use the first conditioning

        if noise_insert == "noise on beginning steps":
            new_conditioning = node_helpers.conditioning_set_values(noisy_embedding, {"start_percent": 0.0, "end_percent": steps_switchover_percent})
            new_conditioning = new_conditioning + node_helpers.conditioning_set_values(conditioning, {"start_percent": steps_switchover_percent, "end_percent": 1.0})
        elif noise_insert == "noise on ending steps":
            new_conditioning = node_helpers.conditioning_set_values(conditioning, {"start_percent": 0.0, "end_percent": steps_switchover_percent})
            new_conditioning = new_conditioning + node_helpers.conditioning_set_values(noisy_embedding, {"start_percent": steps_switchover_percent, "end_percent": 1.0})
        else:
            return (noisy_embedding,)

        return (new_conditioning,)













