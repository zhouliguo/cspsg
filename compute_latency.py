import numpy as np
import torch
from torch.nn import functional as F

from utils.dataloaders import LoadImagesAndLabels_sg
from stream_metrics import StreamSegMetrics
import time
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

def compute_latency_ms_pytorch():
    iterations = 500
    weights = 'runs/train/exp10/weights/m_best.pt'

    device = torch.device('cuda:0')
    model = torch.load(weights, map_location='cpu') # load checkpoint to CPU to avoid CUDA memory leak

    model.eval()
    model = model.to(device)

    with torch.no_grad():
        input = torch.randn((1,3,1024,2048)).to(device, non_blocking=True)
        input.uniform_(0,1)
        elapsed_time = 0
        for i in range(500):
            t_start = time.time()
            outputs = model(input)[0]
            t_end = time.time()
            elapsed_time = elapsed_time+(t_end - t_start)
            print(i)
        latency = elapsed_time / iterations * 1000
    torch.cuda.empty_cache()
    return latency

latency = compute_latency_ms_pytorch()
print("FPS:" + str(1000./latency))
