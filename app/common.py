import modal

_PYTHON_VERSION = "3.12"

app = modal.App(name="modal-batch-transcription")

transcription_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04", add_python=_PYTHON_VERSION
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": "/hf_cache",
            "CXX": "g++",
            "CC": "g++",
        }
    )
    .apt_install("ffmpeg")
    .pip_install(
            "torch==2.7.1",
            "evaluate==0.4.3",
            "librosa==0.11.0",
            "hf_transfer==0.1.9",
            "huggingface_hub[hf-xet]==0.32.4",
            "cuda-python==12.8.0",
            "nemo_toolkit[asr]==2.3.1",
            
        )
    .entrypoint([])
    .add_local_dir("utils", remote_path="/root/utils")
)

data_download_image = (
    modal.Image.debian_slim(python_version=_PYTHON_VERSION)
    .apt_install("ffmpeg")
    .pip_install(
        "datasets[audio]==4.0.0",
        "torch==2.7.1",
        "soundfile==0.13.1"
    )
    .add_local_dir("utils", remote_path="/root/utils")
)

runner_image = (
    modal.Image.debian_slim(python_version=_PYTHON_VERSION)
    .pip_install(
            "pandas==2.3.1",
            "numpy==2.2.6",
            "evaluate==0.4.3",
            "librosa==0.11.0",
            "jiwer==4.0.0",
            "regex==2024.11.6",
            
        )
    .add_local_dir("utils", remote_path="/root/utils")
)

dataset_volume = modal.Volume.from_name("transcription-datasets", create_if_missing=True)
DATASETS_VOLPATH = "/datasets"

model_volume = modal.Volume.from_name("transcription-models", create_if_missing=True)
MODELS_VOLPATH = "/models"

results_volume = modal.Volume.from_name("transcription-results", create_if_missing=True)
RESULTS_VOLPATH = "/results"