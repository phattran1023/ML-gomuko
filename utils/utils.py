import os
import json
import random
import numpy as np
import torch
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Tải cấu hình từ tệp JSON
    
    Args:
        config_path: Đường dẫn đến tệp cấu hình
        
    Returns:
        Dict[str, Any]: Cấu hình đã tải
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config

def set_seed(seed: int) -> None:
    """
    Đặt seed cho tính ngẫu nhiên để tái tạo kết quả
    
    Args:
        seed: Giá trị seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    # Đảm bảo tái tạo được kết quả trên GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
