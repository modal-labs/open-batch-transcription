import modal
import time


volume = modal.Volume.from_name("batch-asr", create_if_missing=True).read_only()
model_volume = modal.Volume.from_name("asr-models", create_if_missing=True)

data_volume_mount = "/data"
model_cache_mount = "/hf_cache"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04", add_python="3.12"
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": model_cache_mount,
            "CXX": "g++",
            "CC": "g++",
        }
    )
    .apt_install("ffmpeg")
    .pip_install(
            "torch",
            "evaluate==0.4.3",
            "librosa==0.11.0",
            "hf_transfer==0.1.9",
            "huggingface_hub[hf-xet]==0.32.4",
            "cuda-python==12.8.0",
            "nemo_toolkit[asr]==2.3.1",
            
        )
    .env(
        {
            "HF_DATASETS_IN_MEMORY_MAX_SIZE": str(16 * 1024 * 1024 * 1024) # 16GB
        }
    )
    .add_local_dir("utils", remote_path="/root/utils")
    
)

app = modal.App(image=image)

@app.cls(
    image=image, 
    gpu="A100-40GB",
    timeout=60*60, 
    volumes={
        data_volume_mount: volume,
        model_cache_mount: model_volume,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
    # _experimental_enable_gpu_snapshot=True,
    # 32 GB memory
    # memory=32768,
    # region='ewr',
    # cloud="oci",
    # max_containers=5
)
class ParakeetEval():
    _MODEL_ID: str = "nvidia/parakeet-tdt-0.6b-v2"
    batch_size: int = modal.parameter(default=128)
    
    @modal.enter()
    def setup(self):
        import os
        import evaluate
        import torch

        from nemo.collections.asr.models import ASRModel
        
        print(f"MODAL_CLOUD_PROVIDER: {os.environ.get('MODAL_CLOUD_PROVIDER')}")
        print(f"MODAL_REGION: {os.environ.get('MODAL_REGION')}")
        print(f"MODAL_ENVIRONMENT: {os.environ.get('MODAL_ENVIRONMENT')}")
        
        self.wer_metric = evaluate.load("wer")

        self.device_obj = torch.device("cuda")
        self.compute_dtype = torch.bfloat16
            
        # Model loading and preparation
        self.asr_model = ASRModel.from_pretrained(
            self._MODEL_ID, 
            map_location=self.device_obj
        )  # type: ASRModel
        self.asr_model.to(self.compute_dtype)
        self.asr_model.eval()

        # Configure decoding strategy
        if self.asr_model.cfg.decoding.strategy != "beam":
            self.asr_model.cfg.decoding.strategy = "greedy_batch"
            self.asr_model.change_decoding_strategy(self.asr_model.cfg.decoding)

    def prepare(self):
        import torch

        self.ds = self.load_data()

        # Warmup
        with torch.autocast("cuda", enabled=False, dtype=self.compute_dtype), torch.inference_mode(), torch.no_grad():
            self.asr_model.transcribe(self.ds["audio"][:4*self.batch_size], batch_size=self.batch_size, verbose=False, num_workers=1)

        

    @modal.method()
    def transcribe(
        self,
        dataset_idx: int,
    ):
        import os
        import datasets
        import torch
        import time
        import evaluate
        from utils import data_utils

        print(f"MODAL_CLOUD_PROVIDER: {os.environ.get('MODAL_CLOUD_PROVIDER')}")
        print(f"MODAL_REGION: {os.environ.get('MODAL_REGION')}")
        print(f"MODAL_ENVIRONMENT: {os.environ.get('MODAL_ENVIRONMENT')}")

        self.dataset_idx = dataset_idx
        self.prepare()

        start_time = time.perf_counter()
        with torch.autocast("cuda", enabled=False, dtype=self.compute_dtype), torch.inference_mode(), torch.no_grad():
            transcriptions = self.asr_model.transcribe(self.ds["audio"], batch_size=self.batch_size, verbose=False, num_workers=1)
        total_time = time.perf_counter() - start_time
        print(f"Total time: {total_time} seconds")

        # Process transcriptions
        if isinstance(transcriptions, tuple) and len(transcriptions) == 2:
            transcriptions = transcriptions[0]
        predictions = [data_utils.normalizer(pred.text) for pred in transcriptions]

        avg_time = total_time / len(transcriptions)
        print(f"Average time: {avg_time} seconds")

        wer = self.wer_metric.compute(references=self.ds["references"], predictions=predictions)
        wer = round(100 * wer, 2)

        audio_length = sum(self.ds["audio_length_s"])
        rtfx = audio_length / total_time
        rtfx = round(rtfx, 2)

        print(f"RTFX: {rtfx}")
        print(f"WER: {wer}%")

        return wer, rtfx, predictions

    def load_data(self):
        import os
        import datasets

        datasets.config.IN_MEMORY_MAX_SIZE = 16 * 1024 * 1024 * 1024

        print(os.environ["MODAL_REGION"])
        print(os.environ["MODAL_CLOUD_PROVIDER"])

        # Load the specific dataset instead of shards
        ds = datasets.load_from_disk(f"{data_volume_mount}/hf-asr-leaderboard-dataset-{self.dataset_idx}")
        return ds

@app.local_entrypoint()
def main():
    parakeet = ParakeetEval()
    for result in parakeet.transcribe.map(range(10)):
        print(result)