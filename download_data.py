import modal
from concurrent.futures import ThreadPoolExecutor
import threading

volume = modal.Volume.from_name("batch-asr", create_if_missing=True)
image = modal.Image.debian_slim().pip_install("datasets","torch","librosa","soundfile")

app = modal.App(image=image)
data_volume_mount = "/data"


def download_dataset(
    dataset_path: str,
    dataset: str,
    split: str,
    # streaming: bool,
    # token: bool,
    # cache_dir: str,
    # keep_in_memory: bool,
) -> None:
    
    from datasets import load_dataset

    # Download and save the dataset locally
    dataset = load_dataset(
        dataset_path,
        dataset,
        split=split,
        num_proc=10
        # streaming=streaming,
        # token=token,
        # cache_dir=cache_dir,
        # keep_in_memory=keep_in_memory,
    )
    return dataset

def split_dataset_by_duration_optimized(dataset, num_containers: int = 10):
    """
    Optimized version that works with batches and avoids individual item access.
    """
    if len(dataset) == 0:
        print("Dataset is empty")
        return []
    
    # Get all audio lengths at once (more efficient than individual access)
    audio_lengths = dataset['audio_length_s']
    total_audio_length = sum(audio_lengths)
    target_length_per_container = total_audio_length / num_containers
    
    print(f"Total audio length: {total_audio_length:.2f} seconds")
    print(f"Target length per container: {target_length_per_container:.2f} seconds")
    
    containers = []
    remaining_indices = list(range(len(dataset)))
    
    for container_idx in range(num_containers):
        if not remaining_indices:
            break
            
        # For the last container, take all remaining indices
        if container_idx == num_containers - 1:
            selected_indices = remaining_indices
        else:
            # Calculate cumulative sum for remaining data
            cumsum = 0
            selected_indices = []
            
            for idx in remaining_indices:
                cumsum += audio_lengths[idx]
                selected_indices.append(idx)
                
                # Stop if we've reached the target length
                if cumsum >= target_length_per_container:
                    break
        
        # Create container dataset
        container_dataset = dataset.select(selected_indices)
        containers.append(container_dataset)
        
        # Calculate actual length for this container
        actual_length = sum([audio_lengths[idx] for idx in selected_indices])
        print(f"Container {container_idx + 1}: {len(container_dataset)} files, {actual_length:.2f} seconds")
        
        # Remove selected indices from remaining
        remaining_indices = [idx for idx in remaining_indices if idx not in selected_indices]
    
    print(f"\nCreated {len(containers)} containers")
    return containers

def save_datasets_parallel(datasets, data_volume_mount: str) -> None:
    """Save multiple datasets in parallel"""
    def save_single_dataset(args):
        i, dataset = args
        dataset.save_to_disk(f"{data_volume_mount}/hf-asr-leaderboard-dataset-{i}")
        print(f"Saved dataset {i} with {len(dataset)} samples")
    
    # Use ThreadPoolExecutor for parallel saving
    with ThreadPoolExecutor(max_workers=min(len(datasets), 4)) as executor:
        executor.map(save_single_dataset, enumerate(datasets))
    
    # Commit and save to the volume
    volume.commit()

@app.function(
        volumes={data_volume_mount: volume}, 
        timeout=3000,
        cpu=10
    )
def download_single_dataset_config(dataset_config):
    """Download a single dataset configuration and ensure it's fully materialized"""
    dataset_name, split = dataset_config
    ds = download_dataset(
        dataset_path="hf-audio/esb-datasets-test-only-sorted",
        dataset=dataset_name,
        split=split,
    )
    
    # Force materialization to avoid cache issues across containers
    # Convert to list and back to ensure data is fully loaded
    print(f"Materializing dataset {dataset_name} ({split}) with {len(ds)} samples...")
    
    # Create a new dataset from the materialized data
    from datasets import Dataset
    materialized_data = []
    for item in ds:
        materialized_data.append(item)
    
    materialized_ds = Dataset.from_list(materialized_data)
    print(f"Materialized dataset {dataset_name} ({split})")
    return materialized_ds

@app.function(
        volumes={data_volume_mount: volume}, 
        timeout=3000,
        cpu=10
    )
def download_datasets_to_modal_sequential(num_containers: int = 10):
    """Sequential version - downloads all datasets in the same container to avoid cache issues"""
    from datasets import concatenate_datasets
    import torch
    
    dataset_configs = [
        ("ami", "test"),
        ("earnings22", "test"),
        ("gigaspeech", "test"),
        ("librispeech", "test.clean"),
        ("librispeech", "test.other"),
        ("spgispeech", "test"),
        ("tedlium", "test"),
        ("voxpopuli", "test"),
    ]
    
    # Download all datasets sequentially in the same container
    print("Downloading datasets sequentially...")
    all_datasets = []
    for dataset_config in dataset_configs:
        ds = download_dataset(
            dataset_path="hf-audio/esb-datasets-test-only-sorted",
            dataset=dataset_config[0],
            split=dataset_config[1],
        )
        all_datasets.append(ds)
        print(f"Downloaded {dataset_config[0]} ({dataset_config[1]}) with {len(ds)} samples")

    print("Concatenating and sorting datasets...")
    combined_dataset = concatenate_datasets(all_datasets)
    combined_dataset = combined_dataset.sort("audio_length_s", reverse=True)
    
    # Skip the heavy data preparation step for now - commented out by user
    def prepare_data(batch):
        batch["filename"] = [audio["path"] for audio in batch["audio"]]
        batch["audio"] = [audio["array"] for audio in batch["audio"]]
        return batch
    combined_dataset = combined_dataset.map(prepare_data, batched=True, batch_size=10, num_proc=10)
    
    print("Splitting dataset into equal-duration subsets...")
    # Split dataset into equal-duration subsets
    dataset_subsets = split_dataset_by_duration_optimized(combined_dataset, num_containers)
    
    print("Saving datasets in parallel...")
    # Save each subset as a separate dataset
    save_datasets_parallel(dataset_subsets, data_volume_mount)

@app.function(
        volumes={data_volume_mount: volume}, 
        timeout=3000,
        cpu=10
    )
def download_datasets_to_modal(num_containers: int = 10):
    """Parallel version with materialized datasets"""
    from datasets import concatenate_datasets
    import torch
    
    dataset_configs = [
        ("ami", "test"),
        ("earnings22", "test"),
        ("gigaspeech", "test"),
        ("librispeech", "test.clean"),
        ("librispeech", "test.other"),
        ("spgispeech", "test"),
        ("tedlium", "test"),
        ("voxpopuli", "test"),
    ]
    
    # Download all datasets in parallel with materialization
    print("Downloading datasets in parallel...")
    all_datasets = []
    for dataset in download_single_dataset_config.map(dataset_configs):
        all_datasets.append(dataset)

    print("Concatenating and sorting datasets...")
    combined_dataset = concatenate_datasets(all_datasets)
    combined_dataset = combined_dataset.sort("audio_length_s", reverse=True)
    
    # Skip the heavy data preparation step for now - commented out by user
    def prepare_data(batch):
        batch["filename"] = [audio["path"] for audio in batch["audio"]]
        batch["audio"] = [audio["array"] for audio in batch["audio"]]
        return batch
    combined_dataset = combined_dataset.map(prepare_data, batched=True, batch_size=10, num_proc=10)
    
    print("Splitting dataset into equal-duration subsets...")
    # Split dataset into equal-duration subsets
    dataset_subsets = split_dataset_by_duration_optimized(combined_dataset, num_containers)
    
    print("Saving datasets in parallel...")
    # Save each subset as a separate dataset
    save_datasets_parallel(dataset_subsets, data_volume_mount)

@app.function(
        volumes={data_volume_mount: volume}, 
        timeout=3000,
        cpu=10
    )
def download_datasets_to_modal_memory_efficient(num_containers: int = 10):
    """Memory-efficient version that processes datasets incrementally"""
    from datasets import concatenate_datasets
    import torch
    
    dataset_configs = [
        ("ami", "test"),
        ("earnings22", "test"),
        ("gigaspeech", "test"),
        ("librispeech", "test.clean"),
        ("librispeech", "test.other"),
        ("spgispeech", "test"),
        ("tedlium", "test"),
        ("voxpopuli", "test"),
    ]
    
    # Download all datasets in parallel
    print("Downloading datasets in parallel...")
    all_datasets = []
    for dataset in download_single_dataset_config.map(dataset_configs):
        all_datasets.append(dataset)

    print("Processing datasets with memory efficiency...")
    
    # Instead of concatenating everything, collect metadata first
    dataset_metadata = []
    for i, ds in enumerate(all_datasets):
        for j, item in enumerate(ds):
            dataset_metadata.append({
                'dataset_idx': i,
                'item_idx': j,
                'audio_length_s': item['audio_length_s'],
                'dataset_name': dataset_configs[i][0],
                'split': dataset_configs[i][1]
            })
    
    # Sort by audio length
    dataset_metadata.sort(key=lambda x: x['audio_length_s'], reverse=True)
    
    # Split metadata into containers
    total_audio_length = sum(item['audio_length_s'] for item in dataset_metadata)
    target_length_per_container = total_audio_length / num_containers
    
    print(f"Total audio length: {total_audio_length:.2f} seconds")
    print(f"Target length per container: {target_length_per_container:.2f} seconds")
    
    containers_metadata = []
    current_container = []
    current_length = 0
    
    for item in dataset_metadata:
        if len(containers_metadata) < num_containers - 1 and current_length + item['audio_length_s'] > target_length_per_container and current_container:
            containers_metadata.append(current_container)
            current_container = []
            current_length = 0
        
        current_container.append(item)
        current_length += item['audio_length_s']
    
    # Add remaining items to last container
    if current_container:
        containers_metadata.append(current_container)
    
    # Create and save datasets one by one
    for container_idx, container_metadata in enumerate(containers_metadata):
        print(f"Creating container {container_idx + 1} with {len(container_metadata)} items...")
        
        # Build the actual dataset for this container
        container_items = []
        for item_meta in container_metadata:
            original_item = all_datasets[item_meta['dataset_idx']][item_meta['item_idx']]
            container_items.append(original_item)
        
        # Create dataset from items
        from datasets import Dataset
        container_dataset = Dataset.from_list(container_items)
        
        # Save immediately to free memory
        container_dataset.save_to_disk(f"{data_volume_mount}/hf-asr-leaderboard-dataset-{container_idx}")
        print(f"Saved container {container_idx + 1} with {len(container_dataset)} samples")
        
        # Clear memory
        del container_dataset
        del container_items
    
    # Commit and save to the volume
    volume.commit()

@app.local_entrypoint()
def main(num_containers: int = 10, use_memory_efficient: bool = False, use_sequential: bool = True):
    if use_memory_efficient:
        download_datasets_to_modal_memory_efficient.remote(num_containers)
    elif use_sequential:
        download_datasets_to_modal_sequential.remote(num_containers)
    else:
        download_datasets_to_modal.remote(num_containers)