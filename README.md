# txt2video-flask
A Flask app for text to video generation


# Installation

First install and setup pytorch with cuda.

Expected nvidia drivers and CUDA and cudnn are already installed and setup. [Help: Link](https://github.com/imxzone/Step-by-Step-Setup-CUDA-cuDNN-and-PyTorch-Installation-on-Windows-with-GPU-Compatibility)

Also expected to have ffmpef installed and path setup. [Help: Link](https://ffmpeg.org/download.html)

### Installation steps:

1) Create a virtual environment and activate it

    `python -m venv venv`

    To activate in linux:
    `source venv/bin/activate`
    To activate in windows:
    `venv\Scripts\activate`

2) Install torch with cuda

    `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124`

3) Install requirements

    `pip install -r requirements.txt`

4) run the flask app

    `flask --app flaskapp --debug run`

    Note: Model [damo-vilab--text-to-video-ms-1.7b](https://huggingface.co/ali-vilab/text-to-video-ms-1.7b/tree/main) of size 3.41GB will get downloaded on first run

Approximate time taken for video generation of 256 x 256 resolution @ 8 frames / second is 01 min 58 sec for a 2sec clip on a system having i5-11400H, 16 GB RAM and RTX 3050 4GB(40W TDP).

Sample generation:

**Prompt** : a clownfish in coral reef

**Result** :

https://github.com/user-attachments/assets/51aa5084-3dcf-4bec-8137-18a76979a288


