import gradio as gr
import infer
import uuid
import threading
from queue import Queue

# ===============================
# JOB QUEUE
# ===============================
job_queue = Queue()
job_status = {}
job_progress = {}
job_result = {}

# ===============================
# WORKER (1 GPU)
# ===============================
def worker():
    while True:
        job_id, params = job_queue.get()
        job_status[job_id] = "running"

        def progress_cb(step, timestep):
            job_progress[job_id] = min(step / params["steps"], 1.0)

        try:
            out = infer.main_infer(
                progress_cb=progress_cb,
                **params
            )
            job_result[job_id] = out
            job_status[job_id] = "done"
        except Exception as e:
            job_status[job_id] = f"error: {e}"

        job_queue.task_done()

threading.Thread(target=worker, daemon=True).start()

# ===============================
# GRADIO FUNCTIONS
# ===============================
def submit_job(ref, vid, prompt, steps, text_cfg, emo_cfg, seed):
    job_id = str(uuid.uuid4())

    job_status[job_id] = "queued"
    job_progress[job_id] = 0.0

    job_queue.put((
        job_id,
        dict(
            ref_image_path=ref,
            driven_video_path=vid,
            prompt_text=prompt,
            steps=int(steps),
            text_cfg=float(text_cfg),
            emo_cfg=float(emo_cfg),
            seed=int(seed),
        )
    ))
    return job_id


def poll_job(job_id):
    if not job_id:
        return 0.0, None

    status = job_status.get(job_id, "unknown")
    progress = job_progress.get(job_id, 0.0)

    if status.startswith("error"):
        raise gr.Error(status)

    if status == "done":
        return progress, job_result[job_id]

    return progress, None


# ===============================
# UI
# ===============================
with gr.Blocks() as demo:
    gr.Markdown("# 🎭 FlashPortrait – Queue Inference (Stable)")

    with gr.Row():
        ref = gr.Image(type="filepath", label="Reference Image")
        vid = gr.Video(label="Driven Video")

    prompt = gr.Textbox(value="The girl is singing")
    steps = gr.Slider(10, 60, 30)
    text_cfg = gr.Slider(0.5, 5.0, 1.0)
    emo_cfg = gr.Slider(1.0, 6.0, 4.0)
    seed = gr.Number(value=42, precision=0)

    submit = gr.Button("🚀 Submit Job")

    job_id_box = gr.Textbox(label="Job ID")

    # ✅ PROGRESS ĐÚNG CHUẨN
    progress_bar = gr.Slider(
        minimum=0.0,
        maximum=1.0,
        value=0.0,
        label="Progress",
        interactive=False,
    )

    output = gr.Video()

    submit.click(
        submit_job,
        inputs=[ref, vid, prompt, steps, text_cfg, emo_cfg, seed],
        outputs=job_id_box,
    )

    # ✅ TIMER ĐÚNG API
    timer = gr.Timer(1.0)
    timer.tick(
        poll_job,
        inputs=job_id_box,
        outputs=[progress_bar, output],
    )

# ===============================
# LAUNCH
# ===============================
demo.launch(
    theme=gr.themes.Soft(primary_hue="indigo"),
    server_name="127.0.0.1",
    server_port=7860,
)
