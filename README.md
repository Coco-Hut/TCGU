# TCGU-torch
This is the PyTorch implementation for our Transferable Condensation Graph Unlearning (TCGU).

## Environment Requirement

Hardware environment: Intel(R) Xeon(R) Silver 4208 CPU, a Quadro RTX 6000 24GB GPU, and 128GB of RAM.

Software environment:Python 3.9.12, Pytorch 1.13.0, and CUDA 11.2.0.

Please refer to PyTorch and PyG to install the environments;

Run 'pip install -r requirements.txt' to download required packages;

## Quick Start

For regular unlearning task (e.g., node unlearning with ratio 0.2 on Cora using GCN)
```bash
python main.py --dataset cora --model GCN --unlearn_task node --unlearn_ratio 0.2 --condensing_loop 1500 --finetune_loop 20 --rank 2 --fs_mode tfs
```
