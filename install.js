module.exports = {
  run: [
    // Edit this step to customize the git repository to use
    {
      method: "shell.run",
      params: {
        message: [
          "git clone https://github.com/Francis-Rings/FlashPortrait app",
        ]
      }
    },
    {
      method: "fs.copy",
      params: {
        src: "infer.py",
        dest: "app\\infer.py"
      }
    },
    {
      method: "fs.copy",
      params: {
        src: "app_gradio.py",
        dest: "app\\app_gradio.py",
      }
    },
    {
      method: "fs.copy",
      params: {
        src: "requirements.txt",
        dest: "app\\requirements.txt",
      }
    },
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",                // Edit this to customize the venv folder path
          path: "app",                // Edit this to customize the path to start the shell from
          // xformers: true   // uncomment this line if your project requires xformers
          // triton: true   // uncomment this line if your project requires triton
          // sageattention: true   // uncomment this line if your project requires sageattention
        }
      }
    },	
    {
      method: "shell.run",
      params: {
        venv: "env",                // Edit this to customize the venv folder path
        path: "app",                // Edit this to customize the path to start the shell from
        message: [
          "uv pip install -r requirements.txt",
          // "uv pip install flash_attn",
          "uv pip install einops",
          "uv pip install gradio",
        ]
      }
    },
    {
      method: "fs.download",
      params: {
        "uri": [
		   "https://huggingface.co/FrancisRing/FlashPortrait/resolve/main/face_det.onnx?download=true",
		   "https://huggingface.co/FrancisRing/FlashPortrait/resolve/main/face_landmark.onnx?download=true",
		   "https://huggingface.co/FrancisRing/FlashPortrait/resolve/main/fast_lora_rank64.safetensors?download=true",
		   "https://huggingface.co/FrancisRing/FlashPortrait/resolve/main/fast_vae.pth?download=true",
		   "https://huggingface.co/FrancisRing/FlashPortrait/resolve/main/pd_fpg.pth?download=true",
		   "https://huggingface.co/FrancisRing/FlashPortrait/resolve/main/portrait_encoder.pt?download=true",
		   "https://huggingface.co/FrancisRing/FlashPortrait/resolve/main/transformer.pt?download=true",
        ],
        "dir": "app/checkpoints/FlashPortrait"
       }
    },	
    {
      method: "fs.download",
      params: {
        "uri": [
		   "https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P/resolve/main/Wan2.1_VAE.pth?download=true",
		   "https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P/raw/main/config.json",
		   "https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P/resolve/main/diffusion_pytorch_model-00001-of-00007.safetensors?download=true",
		   "https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P/resolve/main/diffusion_pytorch_model-00002-of-00007.safetensors?download=true",
		   "https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P/resolve/main/diffusion_pytorch_model-00003-of-00007.safetensors?download=true",
		   "https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P/resolve/main/diffusion_pytorch_model-00004-of-00007.safetensors?download=true",
		   "https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P/resolve/main/diffusion_pytorch_model-00005-of-00007.safetensors?download=true",
		   "https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P/resolve/main/diffusion_pytorch_model-00006-of-00007.safetensors?download=true",
		   "https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P/resolve/main/diffusion_pytorch_model-00007-of-00007.safetensors?download=true",
		   "https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P/raw/main/diffusion_pytorch_model.safetensors.index.json",
		   "https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P/resolve/main/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth?download=true",
		   "https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P/resolve/main/models_t5_umt5-xxl-enc-bf16.pth?download=true",
        ],
        "dir": "app/checkpoints/Wan2.1-I2V-14B-720P"
       }
    },		
    {
      method: "script.start",
      params: {
        uri: "start.js",
        params: {
          venv: "env",
          path: "app",
        }
      }
    },
  ]
}
