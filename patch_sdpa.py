import sys
import mlx_lm

def patch_sdpa_globally(turboquant_sdpa):
    for mod_name, mod in list(sys.modules.items()):
        if mod_name.startswith("mlx_lm.models") and hasattr(mod, "scaled_dot_product_attention"):
            setattr(mod, "scaled_dot_product_attention", turboquant_sdpa)
