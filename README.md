# Batch Transcription

Transcribe speech 100x faster and 100x cheaper with Modal and open models.

### Requirements

- `modal` installed in your current Python virtual environment (`pip install modal`)
- A [Modal](http://modal.com/) account (`modal setup`)
- A Modal token set up in your environment (`modal token new`)

## Models

- **Parakeet TDT 0.6B** (default): Hyperfast, English only transcription
- **Canary models**: Just regular fast multilingual transcription

## Usage

### Download ESB Test Datasets

First stage the data (one-time setup):

```bash
modal run -m run::stage_data
```

This downloads audio files from the HuggingFace ESB test subsets: AMI, Earnings22, GigaSpeech, LibriSpeech (clean/other), SPGISpeech, TEDLIUM, VoxPopuli.

### Run Batch Transcription

```bash
modal run -m run::batch_transcription
```

Or run with arguments:

```bash
modal run -m run::batch_transcription \
  --model_id nvidia/parakeet-tdt-0.6b-v2 \
  --gpu-type L40S \
  --gpu-batch-size 128 \
  --num-requests 25 \
  --job-id my-transcription-job
```

## Configuration Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_id` | `nvidia/parakeet-tdt-0.6b-v2` | NeMo ASR model identifier |
| `--gpu-type` | `L40S` | GPU type for transcription function |
| `--gpu-batch-size` | `128` | Number of audio files per GPU batch |
| `--num-requests` | `25` | Number of parallel Modal function calls |
| `--output-path` | `results` | Path for results directory |
| `--job-id` | Auto-generated if not provided | Job identifier |

## Output

Results are saved to the `asr-results` Modal Volume in two formats:

1. **Summary**: `/results/results_summaries/results_summary_{job_id}.csv`
   - Aggregated metrics (WER, RTFX, timing)
   
2. **Detailed**: `/results/results/{job_id}.csv`
   - Individual transcriptions, ground truth, dataset info

### Metrics

- **WER**: Word Error Rate (%) calculated using normalized text for one request
- **RTFX**: Real-time factor (audio duration / processing time) for one request
- **Total Runtime**: End-to-end job execution time for whole job
