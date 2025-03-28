from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any

class Agent(ABC):
    """
    Lớp cơ sở trừu tượng cho tất cả các agent
    """
    
    def __init__(self, player_id: int):
        """
        Khởi tạo agent
        
        Args:
            player_id: Định danh người chơi (BLACK hoặc WHITE)
        """
        self.player_id = player_id
        
    @abstractmethod
    def get_action(self, game_state: Dict[str, Any]) -> Tuple[int, int]:
        """
        Lấy nước đi từ agent dựa trên trạng thái trò chơi
        
        Args:
            game_state: Trạng thái trò chơi hiện tại
            
        Returns:
            Tuple[int, int]: Tọa độ (x, y) của nước đi được chọn
        """
        pass
    
    def reset(self) -> None:
        """
        Đặt lại trạng thái của agent (nếu có)
        """
        pass
    
    def update(self, game_state: Dict[str, Any], action: Tuple[int, int], reward: float, 
               next_state: Dict[str, Any], done: bool) -> None:
        """
        Cập nhật agent sau mỗi hành động (sử dụng cho các agent học máy)
        
        Args:
            game_state: Trạng thái trò chơi trước khi thực hiện hành động
            action: Hành động đã thực hiện (x, y)
            reward: Phần thưởng nhận được
            next_state: Trạng thái trò chơi sau khi thực hiện hành động
            done: True nếu trò chơi kết thúc, False nếu chưa
        """
        pass
    
    def save(self, filepath: str) -> None:
        """
        Lưu trạng thái của agent (nếu có)
        
        Args:
            filepath: Đường dẫn tới tệp để lưu
        """
        pass
    
    def load(self, filepath: str) -> None:
        """
        Tải trạng thái của agent (nếu có)
        
        Args:
            filepath: Đường dẫn tới tệp để tải
        """
        pass
