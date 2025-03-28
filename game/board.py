import numpy as np
from typing import Tuple, List, Optional

class Board:
    """
    Biểu diễn bàn cờ Gomoku (Caro)
    """
    EMPTY = 0
    BLACK = 1
    WHITE = 2
    
    def __init__(self, size: int = 15):
        """
        Khởi tạo bàn cờ với kích thước được chỉ định
        
        Args:
            size: Kích thước bàn cờ (mặc định là 15x15)
        """
        self.size = size
        self.reset()
        
    def reset(self) -> None:
        """Đặt lại bàn cờ về trạng thái ban đầu"""
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.last_move = None
        
    def place_stone(self, x: int, y: int, stone: int) -> bool:
        """
        Đặt một quân cờ trên bàn
        
        Args:
            x: Tọa độ x
            y: Tọa độ y
            stone: Loại quân cờ (BLACK hoặc WHITE)
            
        Returns:
            bool: True nếu thành công, False nếu không hợp lệ
        """
        if not self.is_valid_move(x, y):
            return False
        
        self.board[x, y] = stone
        self.last_move = (x, y)
        return True
    
    def is_valid_move(self, x: int, y: int) -> bool:
        """
        Kiểm tra nước đi có hợp lệ không
        
        Args:
            x: Tọa độ x
            y: Tọa độ y
            
        Returns:
            bool: True nếu hợp lệ, False nếu không
        """
        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            return False
        return self.board[x, y] == self.EMPTY
    
    def get_valid_moves(self) -> List[Tuple[int, int]]:
        """
        Lấy danh sách các nước đi hợp lệ
        
        Returns:
            List[Tuple[int, int]]: Danh sách các vị trí hợp lệ (x, y)
        """
        valid_moves = []
        for x in range(self.size):
            for y in range(self.size):
                if self.board[x, y] == self.EMPTY:
                    valid_moves.append((x, y))
        return valid_moves
    
    def is_win(self, x: int, y: int, win_length: int = 5) -> bool:
        """
        Kiểm tra xem nước đi tại (x, y) có dẫn đến chiến thắng không
        
        Args:
            x: Tọa độ x
            y: Tọa độ y
            win_length: Số quân cờ liên tiếp để chiến thắng (mặc định là 5)
            
        Returns:
            bool: True nếu chiến thắng, False nếu không
        """
        stone = self.board[x, y]
        if stone == self.EMPTY:
            return False
        
        # Kiểm tra theo 4 hướng: ngang, dọc, chéo xuống, chéo lên
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dx, dy in directions:
            count = 1  # Đã có 1 quân tại vị trí (x, y)
            
            # Kiểm tra theo hai phía của hướng
            for direction in [1, -1]:
                nx, ny = x, y
                for _ in range(win_length - 1):
                    nx += direction * dx
                    ny += direction * dy
                    if (nx < 0 or nx >= self.size or
                        ny < 0 or ny >= self.size or
                        self.board[nx, ny] != stone):
                        break
                    count += 1
            
            if count >= win_length:
                return True
                
        return False
    
    def is_full(self) -> bool:
        """
        Kiểm tra xem bàn cờ đã đầy chưa
        
        Returns:
            bool: True nếu bàn cờ đã đầy, False nếu chưa
        """
        return len(self.get_valid_moves()) == 0
    
    def get_state(self) -> np.ndarray:
        """
        Lấy trạng thái hiện tại của bàn cờ
        
        Returns:
            np.ndarray: Mảng 2D biểu diễn bàn cờ
        """
        return self.board.copy()
    
    def __str__(self) -> str:
        """Biểu diễn bàn cờ dưới dạng chuỗi"""
        symbols = {self.EMPTY: '.', self.BLACK: 'X', self.WHITE: 'O'}
        result = ""
        for i in range(self.size):
            for j in range(self.size):
                result += symbols[self.board[i, j]] + ' '
            result += '\n'
        return result
