import open_clip
import torch


def load_mrl_clip(model_name: str = "ViT-B-32", pretrained: str = "/gscratch/krishna/arnabk1/clip_benchmark/clip_benchmark/models/mrl_true_mp_rank_00_model_states.pt", cache_dir: str = None, device="cuda"):
    model, _, transform = open_clip.create_model_and_transforms(model_name)
    model.load_state_dict(torch.load(pretrained)['module'])
    print("\n...loaded mrl_clip model...\n")
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, transform, tokenizer