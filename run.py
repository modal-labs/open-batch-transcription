import datasets
import tempfile
import argparse
from datetime import datetime

from app.common import (
    app, 
    dataset_volume, 
)
from app.stage_data import download_hf_dataset
from app.transcription import (
    NeMoAsrBatchTranscription,
    TranscriptionRunner,
)
from utils.data import ESB_TEST_SUBSETS, ESB_DATASET_NAME


@app.local_entrypoint()
def stage_data():
    
    prepped_datasets = []
    for data_dict in download_hf_dataset.starmap(ESB_TEST_SUBSETS):
        if data_dict is not None:
            prepped_datasets.append(datasets.Dataset.from_dict(data_dict))

    full_ds = datasets.concatenate_datasets(prepped_datasets).sort("audio_length_s", reverse=True)

    with tempfile.TemporaryFile() as temp_file:
        full_ds.to_csv(temp_file)
        temp_file.seek(0)
        with dataset_volume.batch_upload(force=True) as batch:
            batch.put_file(temp_file, f"/{ESB_DATASET_NAME}/esb_full_features.csv")


@app.local_entrypoint()
def batch_transcription(*args):    

    print("running entrypoint")
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id", 
        type=str, 
        default=NeMoAsrBatchTranscription._DEFAULT_MODEL_ID, 
        help="Model identifier. Should be loadable with NVIDIA NeMo.",
    )
    parser.add_argument(
        "--gpu-type",
        type=str,
        default=NeMoAsrBatchTranscription._DEFAULT_GPU_TYPE,
        help="The GPU type to run the pipeline on.",
    )
    parser.add_argument(
        "--gpu-batch-size", 
        type=int, 
        default=NeMoAsrBatchTranscription._DEFAULT_BATCH_SIZE, 
        help="Number of samples to go through each streamed batch.",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=NeMoAsrBatchTranscription._DEFAULT_NUM_REQUESTS,
        help="Number of calls to make to the run_inference_from_archive method.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="results",
        help="Path to save the combined CSV file",
    )
    parser.add_argument(
        "--job-id",
        type=str,
        default=f"NeMoAsrBatchTranscription_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
        help="Job ID.",
    )
    cfg = parser.parse_args(args=args)

    print("Job Config:")
    print(cfg)
    
    runner = TranscriptionRunner()
    runner.run_transcription.remote(cfg)