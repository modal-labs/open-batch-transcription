import datasets
import soundfile
import numpy as np
import traceback
import os
from app.common import (
    app, data_download_image, dataset_volume, DATASETS_VOLPATH
)
from utils.data import DEFAULT_MAX_THREADS, ESB_DATASETPATH_MODAL

@app.function(
    image=data_download_image,
    volumes={
        DATASETS_VOLPATH: dataset_volume,
    },
    cpu=DEFAULT_MAX_THREADS,
    timeout=60*60,
)
def download_hf_dataset(dataset_path, dataset_name, split):
    

        dataset_path_dest = f"{ESB_DATASETPATH_MODAL}/{dataset_name}/{split}"
        os.makedirs(dataset_path_dest, exist_ok=True)
        
        ds = datasets.load_dataset(dataset_path, dataset_name, split=split)
        def prepare_data(batch):
            filenames = []
            filepaths = []
            durations = []
            
            for audio, id in zip(batch['audio'], batch['id']):
                # # first step added here to make ID and wav filenames unique
                # # several datasets like earnings22 have a hierarchical structure
                # # for eg. earnings22/test/4432298/281.wav, earnings22/test/4450488/281.wav
                # # lhotse uses the filename (281.wav) here as unique ID to create and name cuts
                # # ref: https://github.com/lhotse-speech/lhotse/blob/master/lhotse/dataset/collation.py#L186
                filename = id.replace('/', '_') + ".wav"

                audiofile_path = f"{dataset_path_dest}/{filename}"

                audio_array = np.float32(audio["array"])
                sample_rate = audio["sampling_rate"]
                soundfile.write(audiofile_path, audio_array, sample_rate)

                duration = len(audio_array) / sample_rate

                filenames.append(filename)
                filepaths.append(audiofile_path)
                durations.append(duration) 
                
            batch["filepath"] = filepaths
            batch["filename"] = filenames
            batch["split"] = [split] * len(filenames)
            return batch
        
        try:
            ds = ds.map(
                prepare_data, 
                batched=True, 
                batch_size = len(ds)//DEFAULT_MAX_THREADS, 
                num_proc=DEFAULT_MAX_THREADS, 
                remove_columns="audio"
            )
            ds.to_csv(f"{dataset_path_dest}/features.csv")
            return ds.to_dict()
        except Exception as e:
            print(f"Error downloading {dataset_name} {split}: {e}")
            print(traceback.format_exc())
            return None