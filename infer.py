# ============================================================
# FlashPortrait - Refactored Infer (Stable for Gradio / Queue)
# ============================================================

import os
import sys
import cv2
import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from diffusers import FlowMatchEulerDiscreteScheduler
from einops import rearrange

# ============================================================
# PATH FIX
# ============================================================
current_file_path = os.path.abspath(__file__)
project_roots = [
    os.path.dirname(current_file_path),
    os.path.dirname(os.path.dirname(current_file_path)),
    os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))),
]
for project_root in project_roots:
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

# ============================================================
# WAN IMPORTS (GIỮ NGUYÊN)
# ============================================================
from wan.dist import set_multi_gpus_devices, shard_model
from wan.models import (
    AutoencoderKLWan,
    CLIPModel,
    WanT5EncoderModel,
    WanTransformer3DModel,
)
from wan.models.cache_utils import get_teacache_coefficients
from wan.models.face_align import FaceAlignment
from wan.models.face_model import FaceModel
from wan.models.pdf import FanEncoder, det_landmarks, get_drive_expression_pd_fgc
from wan.models.portrait_encoder import PortraitEncoder
from wan.pipeline.pipeline_wan_long import WanI2VLongPipeline
from wan.utils.fp8_optimization import (
    convert_model_weight_to_float8,
    replace_parameters_by_name,
    convert_weight_dtype_wrapper,
)
from wan.utils.lora_utils import merge_lora, unmerge_lora
from wan.utils.utils import (
    filter_kwargs,
    save_videos_grid,
    simple_save_videos_grid,
)
from wan.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from xfuser.core.distributed import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
    get_world_group,
    init_distributed_environment,
    initialize_model_parallel,
)
import torch.distributed as dist

# ============================================================
# GLOBAL CONFIG (GIỮ NGUYÊN TỪ FILE GỐC)
# ============================================================
GPU_memory_mode = "model_full_load"
ulysses_degree = 1
ring_degree = 1
fsdp_dit = True
fsdp_text_encoder = True
compile_dit = False

enable_teacache = False
teacache_threshold = 0.10
num_skip_start_steps = 5
teacache_offload = False
cfg_skip_ratio = 0

enable_riflex = False
riflex_k = 6

config_path = "config/wan2.1/wan_civitai.yaml"
wan_model_name = "/path/FlashPortrait/checkpoints/Wan2.1-I2V-14B-720P"

sampler_name = "Flow"
shift = 5

transformer_path = "/path/FlashPortrait/checkpoints/FlashPortrait/transformer.pt"
portrait_encoder_path = "/path/FlashPortrait/checkpoints/FlashPortrait/portrait_encoder.pt"
det_model_path = "/path/FlashPortrait/checkpoints/FlashPortrait/face_det.onnx"
alignment_model_path = "/path/FlashPortrait/checkpoints/FlashPortrait/face_landmark.onnx"
pd_fpg_model_path = "/path/FlashPortrait/checkpoints/FlashPortrait/pd_fpg.pth"

sample_size = [512, 512]
max_size = 720
sub_num_frames = 201
latents_num_frames = 51
context_overlap = 30
context_size = 51
ip_scale = 1.0
text_cfg_scale = 1.0
emo_cfg_scale = 4.0
fps = 25

weight_dtype = torch.bfloat16

negative_prompt = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体"
)

save_path = "samples/wan-videos-i2v"

# ============================================================
# DEVICE FIX (🔥 QUAN TRỌNG – FIX LỖI CỦA BẠN)
# ============================================================
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# ============================================================
# LOAD CONFIG
# ============================================================
config = OmegaConf.load(config_path)

# ============================================================
# LOAD MODELS (GIỮ NGUYÊN LOGIC)
# ============================================================
transformer = WanTransformer3DModel.from_pretrained(
    os.path.join(wan_model_name, config["transformer_additional_kwargs"].get("transformer_subpath", "transformer")),
    transformer_additional_kwargs=OmegaConf.to_container(config["transformer_additional_kwargs"]),
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
)

state_dict = torch.load(transformer_path, map_location="cpu")
state_dict = state_dict.get("state_dict", state_dict)
transformer.load_state_dict(state_dict, strict=False)

vae = AutoencoderKLWan.from_pretrained(
    os.path.join(wan_model_name, config["vae_kwargs"].get("vae_subpath", "vae")),
    additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
).to(weight_dtype)

tokenizer = AutoTokenizer.from_pretrained(
    os.path.join(wan_model_name, config["text_encoder_kwargs"].get("tokenizer_subpath", "tokenizer"))
)

text_encoder = WanT5EncoderModel.from_pretrained(
    os.path.join(wan_model_name, config["text_encoder_kwargs"].get("text_encoder_subpath", "text_encoder")),
    additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]),
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
).eval()

clip_image_encoder = CLIPModel.from_pretrained(
    os.path.join(wan_model_name, config["image_encoder_kwargs"].get("image_encoder_subpath", "image_encoder"))
).eval()

face_aligner = FaceModel(
    face_alignment_module=FaceAlignment(
        gpu_id=None,
        alignment_model_path=alignment_model_path,
        det_model_path=det_model_path,
    ),
    reset=False,
)

pd_fpg_motion = FanEncoder()
pd_fpg_motion.load_state_dict(torch.load(pd_fpg_model_path, map_location="cpu"), strict=False)
pd_fpg_motion = pd_fpg_motion.eval()

portrait_encoder = PortraitEncoder(adapter_in_dim=768, adapter_proj_dim=2048)
portrait_encoder.load_state_dict(torch.load(portrait_encoder_path, map_location="cpu"), strict=False)
portrait_encoder = portrait_encoder.eval()

scheduler = FlowMatchEulerDiscreteScheduler(
    **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config["scheduler_kwargs"]))
)

pipeline = WanI2VLongPipeline(
    transformer=transformer,
    vae=vae,
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    scheduler=scheduler,
    clip_image_encoder=clip_image_encoder,
    portrait_encoder=portrait_encoder,
).to(device)

# ============================================================
# PROGRESS CALLBACK (STEP THẬT)
# ============================================================
_PROGRESS_CB = None

def register_progress_callback(cb):
    global _PROGRESS_CB
    _PROGRESS_CB = cb

def step_callback(step: int, timestep: int, latents):
    if _PROGRESS_CB:
        _PROGRESS_CB(step, timestep)

# ============================================================
# EMOTION FEATURE
# ============================================================
def get_emo_feature(video_path):
    pd_fpg_motion.to(device)
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    landmarks = det_landmarks(face_aligner, frames)[1]
    emo_list = get_drive_expression_pd_fgc(pd_fpg_motion, frames, landmarks, device)

    emo_feat, head_emo_feat = [], []
    for emo in emo_list:
        emo_f = torch.cat([emo["eye_embed"], emo["emo_embed"], emo["mouth_feat"]], dim=1)
        head_emo_feat.append(torch.cat([emo["headpose_emb"], emo_f], dim=1))
        emo_feat.append(emo_f)

    return (
        torch.cat(emo_feat),
        torch.cat(head_emo_feat),
        cap.get(cv2.CAP_PROP_FPS),
        len(frames),
    )

# ============================================================
# MAIN INFERENCE (DÙNG CHO GRADIO / QUEUE)
# ============================================================
def main_infer(
    ref_image_path,
    driven_video_path,
    prompt_text,
    seed=42,
    steps=30,
    text_cfg=1.0,
    emo_cfg=4.0,
    progress_cb=None,
):
    if progress_cb:
        register_progress_callback(progress_cb)

    generator = torch.Generator(device=device).manual_seed(seed)

    image = Image.open(ref_image_path).convert("RGB")
    w, h = image.size
    scale = max_size / max(w, h)
    w, h = int(w * scale) // 16 * 16, int(h * scale) // 16 * 16
    image = image.resize((w, h), Image.LANCZOS)

    input_video = torch.from_numpy(np.array(image)).permute(2, 0, 1).unsqueeze(0).unsqueeze(2) / 255
    input_video = input_video.repeat(1, 1, sub_num_frames, 1, 1)
    mask_video = torch.zeros_like(input_video[:, :1])
    mask_video[:, :, 1:] = 255

    emo_feat, head_emo_feat, fps, num_frames = get_emo_feature(driven_video_path)

    with torch.no_grad():
        sample = pipeline(
            prompt_text,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            height=h,
            width=w,
            generator=generator,
            num_inference_steps=steps,
            guidance_scale=4.0,
            video=input_video,
            mask_video=mask_video,
            clip_image=image,
            head_emo_feat_all=head_emo_feat.unsqueeze(0).to(device),
            text_cfg_scale=text_cfg,
            emo_cfg_scale=emo_cfg,
            callback=step_callback,
            callback_steps=1,
        ).videos

    os.makedirs(save_path, exist_ok=True)
    out_path = os.path.join(save_path, f"{len(os.listdir(save_path))+1:08d}.mp4")
    simple_save_videos_grid(sample[:, :, 1:], out_path, fps=fps)
    return out_path
