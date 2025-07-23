import modal
from pathlib import Path
import pandas as pd
import time
import logging

from app.common import (
    app, 
    transcription_image, 
    runner_image,
    dataset_volume, 
    model_volume, 
    results_volume,
    DATASETS_VOLPATH, 
    MODELS_VOLPATH, 
    RESULTS_VOLPATH
)
from utils.data import ESB_DATASET_NAME, ESB_DATASETPATH_MODAL

MINUTES = 60 # seconds

with transcription_image.imports():
    import nemo.collections.asr as nemo_asr
    import torch
    import evaluate
    import utils.normalizer.data_utils as du
    import utils.normalizer.eval_utils as eu
    from pathlib import Path
    import time
    from utils.data import copy_concurrent




@app.cls(
    image=transcription_image, 
    timeout=60*MINUTES, 
    volumes={
        DATASETS_VOLPATH: dataset_volume,
        MODELS_VOLPATH: model_volume,
        RESULTS_VOLPATH: results_volume,
    },
    scaledown_window=5,
)
class NeMoAsrBatchTranscription():
    DEFAULT_MODEL_ID = "nvidia/parakeet-tdt-0.6b-v2"
    DEFAULT_GPU_TYPE = "L40S"
    DEFAULT_BATCH_SIZE = 128
    DEFAULT_NUM_REQUESTS = 25
    model_id: str = modal.parameter(default=DEFAULT_MODEL_ID)
    gpu_batch_size: int = modal.parameter(default=DEFAULT_BATCH_SIZE)
    
    
    @modal.enter()
    def setup(self):

        self._COMPUTE_DTYPE = torch.bfloat16

        # silence chatty logs from nemo
        logging.getLogger("nemo_logger").setLevel(logging.CRITICAL)

        self.asr_model = nemo_asr.models.ASRModel.from_pretrained(self.model_id)         
        self.asr_model.to(self._COMPUTE_DTYPE)
        self.asr_model.eval()

        # Configure decoding strategy
        if self.asr_model.cfg.decoding.strategy != "beam":
            self.asr_model.cfg.decoding.strategy = "greedy_batch"
            self.asr_model.change_decoding_strategy(self.asr_model.cfg.decoding)

    @modal.method()
    async def run_inference(self, audio_filepaths):

        local_filepaths = [path.replace(ESB_DATASETPATH_MODAL, '/tmp') for path in audio_filepaths]
        filenames = [filepath.split('/')[-1] for filepath in local_filepaths]

        copy_concurrent(Path(ESB_DATASETPATH_MODAL), Path('/tmp/'), filenames)
        
        start_time = time.perf_counter()
        with torch.autocast("cuda", enabled=False, dtype=self._COMPUTE_DTYPE), torch.inference_mode(), torch.no_grad():
            if 'canary' in self.model_id:
                transcriptions = self.asr_model.transcribe(local_filepaths, batch_size=self.gpu_batch_size, verbose=False, pnc='no', num_workers=1)
            else:
                transcriptions = self.asr_model.transcribe(local_filepaths, batch_size=self.gpu_batch_size, num_workers=1)

        total_time = time.perf_counter() - start_time
        print("Total time:", total_time)
        
        # Process transcriptions
        if isinstance(transcriptions, tuple) and len(transcriptions) == 2:
            transcriptions = transcriptions[0]
        predictions = [pred.text for pred in transcriptions]    

        return {
            "num_samples": len(filenames),
            "transcriptions": predictions,
            "total_time": total_time,
        }
    

with runner_image.imports():
    import evaluate
    import utils.normalizer.data_utils as du
    import utils.normalizer.eval_utils as eu
    from utils.data import distribute_audio

@app.cls(
    image=runner_image,
    timeout=60*60,
    volumes={
        DATASETS_VOLPATH: dataset_volume,
        RESULTS_VOLPATH: results_volume,
    },
)
class TranscriptionRunner():
    num_requests: int = modal.parameter()

    @modal.method()
    def run_transcription(self, cfg):

        print(f"Starting transcription job: {cfg.job_id}...")
        start_time = time.perf_counter()
        batch_start_creation_time = time.perf_counter()
        data_df = pd.read_csv(f"/datasets/{ESB_DATASET_NAME}/esb_full_features.csv")
        dfs = distribute_audio(data_df, self.num_requests)
        batch_creation_time = time.perf_counter() - batch_start_creation_time
        print(f"Batch creation time: {loading_time} seconds")

        results = []
        

        nemo_transcription = NeMoAsrBatchTranscription.with_options(
            gpu=cfg.gpu_type,
        )(
            model_id=cfg.model_id, 
            gpu_batch_size=cfg.gpu_batch_size,
        )

        print("Running inference...")
        for result in nemo_transcription.run_inference.map([df['filepath'].tolist() for df in dfs]):
            results.append(result)
        
        total_runtime = time.perf_counter() - start_time    
        print(f"Total runtime: {total_runtime} seconds")

        for result, df in zip(results, dfs):
            result['total_runtime'] = total_runtime
            result['job_id'] = cfg.job_id
            result['audio_length_s'] = df['audio_length_s'].tolist()
            result['original_text'] = df['text'].tolist()
            result['dataset'] = df['dataset'].tolist()
            result['split'] = df['split'].tolist()

        print("Scoring results...")
        scored_results = []
        for scored_result in self.score_call.map(results):
            scored_results.append(scored_result)

        self.save_results(scored_results, cfg)

        print(f"Transcription job {cfg.job_id} complete.")
    
    @modal.method()
    def score_call(self, results):
        wer_metric = evaluate.load("wer")
        
        # Calculate metrics
        normalized_predictions = [du.normalizer(pred) for pred in results['transcriptions']]
        normalized_references = [du.normalizer(ref) for ref in results['original_text']]
        wer = wer_metric.compute(references=normalized_references, predictions=normalized_predictions)
        wer = round(100 * wer, 2)

        audio_length = sum(results['audio_length_s'])
        rtfx = audio_length / results['total_time']
        rtfx = round(rtfx, 2)

        results['wer'] = wer
        results['rtfx'] = rtfx
        results['total_audio_length'] = audio_length
        
        return results
    
    def save_results(self, results, cfg):
        # save the results to a csv
        results_df = pd.DataFrame(results)

        results_summary_dir = Path(f"{RESULTS_VOLPATH}/results_summaries")
        results_summary_dir.mkdir(parents=True, exist_ok=True)
        results_df.drop(columns=['transcriptions', 'original_text', 'audio_length_s', 'dataset', 'split'],inplace=False).to_csv(results_summary_dir / f"results_summary_{cfg.job_id}.csv", index=False)

        results_dir = Path(f"{RESULTS_VOLPATH}/results")
        results_dir.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(results_dir / f"results_{cfg.job_id}.csv", index=False)

    

        
        
    





