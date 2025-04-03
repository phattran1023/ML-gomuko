import os
import random
import torch
from typing import List, Optional
from agents.rl_agent import RLAgent,Board

class OpponentPool:
    """Quản lý một nhóm các mô hình đối thủ từ các checkpoint trước đó"""
    
    def __init__(self, max_size: int = 5):
        """
        Khởi tạo opponent pool
        
        Args:
            max_size: Số lượng mô hình tối đa trong pool
        """
        self.max_size = max_size
        self.opponents = []  # Danh sách các đường dẫn đến mô hình
    
    def add_model(self, model_path: str) -> None:
        """
        Thêm một mô hình vào pool
        
        Args:
            model_path: Đường dẫn đến mô hình cần thêm
        """
        if model_path not in self.opponents:
            self.opponents.append(model_path)
            
            # Nếu vượt quá kích thước tối đa, loại bỏ mô hình cũ nhất
            if len(self.opponents) > self.max_size:
                self.opponents.pop(0)
    
    def get_random_opponent(self, board_size: int) -> Optional[RLAgent]:
        """
        Lấy một đối thủ ngẫu nhiên từ pool
        
        Args:
            board_size: Kích thước bàn cờ
            
        Returns:
            RLAgent: Agent đối thủ hoặc None nếu pool rỗng
        """
        if not self.opponents:
            return None
            
        model_path = random.choice(self.opponents)
        if not os.path.exists(model_path):
            # Nếu mô hình không tồn tại, xóa khỏi pool
            self.opponents.remove(model_path)
            return self.get_random_opponent(board_size)
            
        opponent = RLAgent(Board.WHITE, board_size=board_size)
        opponent.load(model_path)
        opponent.epsilon = 0.1  # Tăng từ 0.05 lên 0.1 để thêm yếu tố ngẫu nhiên
        
        return opponent
