from typing import Tuple, Optional, Dict, Any
from .board import Board

class GomokuGame:
    """
    Quản lý luật chơi và trạng thái của trò chơi Gomoku (Caro)
    """
    
    def __init__(self, board_size: int = 15, win_length: int = 5):
        """
        Khởi tạo trò chơi Gomoku
        
        Args:
            board_size: Kích thước bàn cờ (mặc định 15x15)
            win_length: Số quân cờ liên tiếp để chiến thắng (mặc định 5)
        """
        self.board = Board(board_size)
        self.win_length = win_length
        self.current_player = Board.BLACK
        self.game_over = False
        self.winner = None
        
    def reset(self) -> None:
        """Đặt lại trò chơi về trạng thái ban đầu"""
        self.board.reset()
        self.current_player = Board.BLACK
        self.game_over = False
        self.winner = None
        
    def make_move(self, x: int, y: int) -> bool:
        """
        Thực hiện một nước đi
        
        Args:
            x: Tọa độ x
            y: Tọa độ y
            
        Returns:
            bool: True nếu nước đi hợp lệ, False nếu không
        """
        if self.game_over or not self.board.is_valid_move(x, y):
            return False
            
        self.board.place_stone(x, y, self.current_player)
        
        # Kiểm tra chiến thắng
        if self.board.is_win(x, y, self.win_length):
            self.game_over = True
            self.winner = self.current_player
            return True
            
        # Kiểm tra hòa
        if self.board.is_full():
            self.game_over = True
            self.winner = None
            return True
            
        # Chuyển lượt
        self.current_player = Board.WHITE if self.current_player == Board.BLACK else Board.BLACK
        return True
        
    def get_state(self) -> Dict[str, Any]:
        """
        Lấy trạng thái hiện tại của trò chơi
        
        Returns:
            Dict: Trạng thái trò chơi hiện tại
        """
        return {
            'board': self.board.get_state(),
            'current_player': self.current_player,
            'game_over': self.game_over,
            'winner': self.winner,
            'valid_moves': self.board.get_valid_moves()
        }
    
    def is_game_over(self) -> bool:
        """
        Kiểm tra xem trò chơi đã kết thúc chưa
        
        Returns:
            bool: True nếu trò chơi kết thúc, False nếu chưa
        """
        return self.game_over
    
    def get_winner(self) -> Optional[int]:
        """
        Lấy người chiến thắng
        
        Returns:
            Optional[int]: BLACK/WHITE nếu có người thắng, None nếu hòa hoặc chưa kết thúc
        """
        return self.winner if self.game_over else None
    
    def get_current_player(self) -> int:
        """
        Lấy người chơi hiện tại
        
        Returns:
            int: BLACK hoặc WHITE
        """
        return self.current_player
    
    def clone(self) -> 'GomokuGame':
        """
        Tạo bản sao của trạng thái trò chơi hiện tại
        
        Returns:
            GomokuGame: Bản sao của trò chơi hiện tại
        """
        clone = GomokuGame(self.board.size, self.win_length)
        clone.board.board = self.board.get_state()
        clone.board.last_move = self.board.last_move
        clone.current_player = self.current_player
        clone.game_over = self.game_over
        clone.winner = self.winner
        return clone
