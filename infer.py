# ===============================
# FLASH PORTRAIT - REFACTORED INFER
# ===============================

import os, sys, cv2, torch, numpy as np
from PIL import Image
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from diffusers import FlowMatchEulerDiscreteScheduler
from einops import rearrange

# ===============================
# LOAD ORIGINAL DEPENDENCIES
# (giữ nguyên như file gốc)
# ===============================
# ⚠️ Toàn bộ phần import + load model
# GIỮ NGUYÊN từ file infer.py gốc của bạn
# 👉 MÌNH CHỈ LƯỢC BỎ Ở ĐÂY ĐỂ ĐỠ DÀI
# 👉 BẠN COPY NGUYÊN KHỐI LOAD MODEL CỦA FILE CŨ LÊN ĐÂY
# ===============================

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
# MAIN INFERENCE FUNCTION
# ============================================================
def main_infer(
    ref_image_path: str,
    driven_video_path: str,
    prompt_text: str,
    seed: int = 42,
    steps: int = 30,
    text_cfg: float = 1.0,
    emo_cfg: float = 4.0,
    save_dir: str = "samples/wan-videos-i2v",
    progress_cb=None,
):
    """
    Inference 1 job – KHÔNG reload model
    """

    if progress_cb:
        register_progress_callback(progress_cb)

    os.makedirs(save_dir, exist_ok=True)

    generator = torch.Generator(device=device).manual_seed(seed)

    # ===============================
    # PREPROCESS IMAGE
    # ===============================
    image_start = Image.open(ref_image_path).convert("RGB")
    clip_image = image_start.copy()

    width, height = image_start.size
    scale = max_size / max(width, height)
    width, height = int(width * scale), int(height * scale)
    width = width // 16 * 16
    height = height // 16 * 16

    image_start = image_start.resize((width, height), Image.LANCZOS)
    clip_image = clip_image.resize((width, height), Image.LANCZOS)

    input_video = torch.from_numpy(np.array(image_start)) \
        .permute(2, 0, 1).unsqueeze(0).unsqueeze(2) / 255.0

    input_video = input_video.repeat(1, 1, sub_num_frames, 1, 1)
    input_video_mask = torch.zeros_like(input_video[:, :1])
    input_video_mask[:, :, 1:] = 255

    # ===============================
    # EMOTION FEATURE
    # ===============================
    emo_feat_all, head_emo_feat_all, fps, num_frames = get_emo_feature(
        driven_video_path, face_aligner, pd_fpg_motion, device=device
    )

    emo_feat_all = emo_feat_all.unsqueeze(0)
    head_emo_feat_all = head_emo_feat_all.unsqueeze(0)

    # ===============================
    # PIPELINE CALL (HOOK PROGRESS)
    # ===============================
    with torch.no_grad():
        sample = pipeline(
            prompt_text,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            generator=generator,
            num_inference_steps=steps,
            guidance_scale=4.0,
            video=input_video,
            mask_video=input_video_mask,
            clip_image=clip_image,
            head_emo_feat_all=head_emo_feat_all.to(device),
            sub_num_frames=sub_num_frames,
            latents_num_frames=latents_num_frames,
            context_overlap=context_overlap,
            context_size=context_size,
            ip_scale=ip_scale,
            text_cfg_scale=text_cfg,
            emo_cfg_scale=emo_cfg,
            callback=step_callback,
            callback_steps=1,   # 🔥 STEP THẬT
        ).videos

    # ===============================
    # SAVE VIDEO
    # ===============================
    index = len(os.listdir(save_dir)) + 1
    out_path = os.path.join(save_dir, f"{index:08d}.mp4")
    simple_save_videos_grid(sample[:, :, 1:], out_path, fps=fps)

    return out_path
