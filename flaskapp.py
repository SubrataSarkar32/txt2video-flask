from flask import Flask, request, render_template, send_file
from diffusers import TextToVideoSDPipeline
import torch
import os
import uuid
import imageio
import gc


app = Flask(__name__)
output_dir = "static/videos"
os.makedirs(output_dir, exist_ok=True)


def make_pipeline_generator(
    device: str, cpu_offload: bool, attention_slice: bool
) -> TextToVideoSDPipeline:
    """Create text2video pipeline"""
    pipeline = TextToVideoSDPipeline.from_pretrained(
        "damo-vilab/text-to-video-ms-1.7b",
        cache_dir="./cache",
        variant="fp16",
        torch_dtype=torch.float32 if device == "cpu" else torch.float16,
    )

    if cpu_offload:
        pipeline.enable_sequential_cpu_offload()
    else:
        pipeline = pipeline.to(torch.device(device))

    if attention_slice:
        pipeline.enable_attention_slicing()

    return pipeline


def generate(
    prompt: str,
    num_frames: int,
    num_steps: int,
    seed: int,
    height: int,
    width: int,
    device: str,
    cpu_offload: bool,
    attention_slice: bool,
) -> list:
    """Generate video with text2video pipeline"""
    pipeline = make_pipeline_generator(
        device=device, cpu_offload=cpu_offload, attention_slice=attention_slice
    )
    generator = torch.Generator().manual_seed(seed)
    video = pipeline(
        prompt=prompt,
        num_frames=num_frames,
        num_inference_steps=num_steps,
        height=height,
        width=width,
        generator=generator,
    ).frames
    torch.cuda.empty_cache()
    gc.collect()
    return video

# Load the Hugging Face text-to-video model
# device = "cuda" if torch.cuda.is_available() else "cpu"
# video_pipeline = TextToVideoSDPipeline.from_pretrained("ali-vilab/modelscope-damo-text-to-video-synthesis", torch_dtype=torch.float16)
# video_pipeline.to(device)

@app.route('/', methods=['GET', 'POST'])
def index():
    video_path = None

    if request.method == 'POST':
        text = request.form.get('text')
        if text:
            filename = f"{uuid.uuid4()}.mp4"
            video_output_path = os.path.join(output_dir, filename)

            # Generate video from text
            with torch.no_grad():
                video_frames = generate(text, num_steps=50, seed=44, height=256, width=256, num_frames=16, device="cuda", cpu_offload=True, attention_slice=False)
            
            # Save video
            save_video(video_frames, video_output_path)
            video_path = f"/{video_output_path}"

    return render_template('index.html', video_path=video_path)

@app.route('/download/<filename>')
def download(filename):
    return send_file(os.path.join(output_dir, filename), as_attachment=True)

def save_video(frames, path):
    f, h, w, c = frames[0].shape
    frames = frames.reshape(-1, h, w, c)
    imageio.mimsave(path, frames, fps=8)


# if __name__ == '__main__':
#     app.run(debug=True)
