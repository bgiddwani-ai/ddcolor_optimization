# DDColor Triton Deployment

Quick setup guide for deploying DDColor image colorization model with NVIDIA Triton Inference Server.

## Prerequisites

- NVIDIA GPU (device 1 used in examples)
- Docker with GPU support
- Git and wget

## Setup Steps

### 1. Clone Repository and Download Model

```bash
# Clone DDColor repository
git clone https://github.com/piddnad/DDColor.git

# Download pre-trained model weights
wget https://huggingface.co/piddnad/DDColor-models/resolve/main/ddcolor_paper_tiny.pth \
  -P /mnt/raid0/bharat/customer_engg/ddcolor/DDColor/pretrain
```

### 2. Model Development (PyTorch)

Run the lab tutorials to create your Triton model repository:

```bash
docker run -it --rm --gpus '"device=1"' \
  -v $PWD:/workspace \
  --net=host --ipc=host \
  nvcr.io/nvidia/pytorch:24.12-py3
```

### 3. Run the Tutorial Notebooks:

Follow the sequence

### 4. Deploy with Triton Server

Once the model repository is created:

```bash
# Start Triton container
docker run -it --rm --gpus '"device=1"' \
  -v $PWD:/workspace \
  --net=host --ipc=host \
  nvcr.io/nvidia/tritonserver:24.12-py3

# Install dependencies
pip install opencv-python==4.10.0.82
apt update
apt-get install ffmpeg libsm6 libxext6 -y

# Start Triton Server
tritonserver --model-repository=/workspace/model_repository_trt \
  --log-verbose=1 \
  --allow-metrics=true
```

## Project Structure

```
.
├── DDColor/                    # Cloned repository
│   └── pretrain/              # Model weights
│       └── ddcolor_paper_tiny.pth
└── model_repository_trt/      # Triton model repository (created during setup)
```

## Notes

- GPU device 1 is specified in all commands
- Host networking and IPC are enabled for optimal performance
- Metrics are enabled for monitoring
- Verbose logging helps with debugging

## References

- [DDColor GitHub](https://github.com/piddnad/DDColor)
- [DDColor Model Weights](https://huggingface.co/piddnad/DDColor-models)
- [NVIDIA Triton Documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/)