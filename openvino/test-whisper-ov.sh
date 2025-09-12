mkdir whisper-test
cd whisper-test

python3 -m venv ov-genai-env # Windows: python -m venv ov-genai-env
source ov-genai-env/bin/activate # Windows: ov-genai-env\Scripts\activate

pip install openvino-genai librosa

sudo apt-get install git-lfs # For Windows, download and install from https://git-lfs.github.com/
git lfs install
git clone https://huggingface.co/OpenVINO/whisper-small-fp16-ov

wget https://raw.githubusercontent.com/openvinotoolkit/openvino.genai/refs/heads/master/samples/python/whisper_speech_recognition/whisper_speech_recognition.py
wget https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/courtroom.wav

# Usage: python whisper_speech_recognition.py <model_dir> <wav_file_path> [device]
python whisper_speech_recognition.py whisper-small-fp16-ov courtroom.wav NPU
