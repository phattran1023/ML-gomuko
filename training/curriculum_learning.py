import numpy as np
from typing import Tuple, Dict, Any
from game.board import Board
from game.game import GomokuGame

class CurriculumGenerator:
    """
    Tạo tình huống huấn luyện theo độ khó tăng dần
    """
    def __init__(self, board_size: int = 15):
        self.board_size = board_size
        self.stage = 0
        self.max_stage = 5
        
    def get_curriculum_game(self, episode: int, total_episodes: int) -> GomokuGame:
        """
        Tạo ra một trò chơi với tình huống phù hợp với giai đoạn hiện tại
        
        Args:
            episode: Tập huấn luyện hiện tại
            total_episodes: Tổng số tập huấn luyện
            
        Returns:
            GomokuGame: Trò chơi đã được thiết lập với tình huống phù hợp
        """
        # Cập nhật stage dựa trên tiến trình huấn luyện
        progress = episode / total_episodes
        self.stage = min(int(progress * self.max_stage), self.max_stage - 1)
        
        game = GomokuGame(self.board_size)
        
        if self.stage == 0:
            # Giai đoạn 1: Bàn cờ trống, học từ đầu
            return game
        elif self.stage == 1:
            # Giai đoạn 2: Bắt đầu với một vài quân cờ ngẫu nhiên
            return self._create_random_start_game(3, 5)
        elif self.stage == 2:
            # Giai đoạn 3: Bắt đầu với nhiều quân cờ hơn
            return self._create_random_start_game(6, 10)
        elif self.stage == 3:
            # Giai đoạn 4: Tạo tình huống có thể tấn công
            return self._create_attack_position()
        elif self.stage >= 4:
            # Giai đoạn 5: Tạo tình huống phòng thủ hoặc tình huống phức tạp
            return self._create_defense_position()
        
        return game
    
    def _create_random_start_game(self, min_moves: int, max_moves: int) -> GomokuGame:
        """Tạo trò chơi với một số nước đi ngẫu nhiên ban đầu"""
        game = GomokuGame(self.board_size)
        
        num_moves = np.random.randint(min_moves, max_moves + 1)
        for _ in range(num_moves):
            if game.is_game_over():
                break
                
            valid_moves = game.get_state()['valid_moves']
            if not valid_moves:
                break
                
            move_idx = np.random.randint(0, len(valid_moves))
            x, y = valid_moves[move_idx]
            game.make_move(x, y)
        
        return game
    
    def _create_attack_position(self) -> GomokuGame:
        """Tạo tình huống tấn công (có cơ hội tạo 3 hoặc 4 quân liên tiếp)"""
        game = GomokuGame(self.board_size)
        
        # Đặt quân đen thành một hàng 3 quân
        center_x, center_y = self.board_size // 2, self.board_size // 2
        
        # 50% cơ hội chọn hàng ngang hoặc dọc
        if np.random.random() < 0.5:  # Hàng ngang
            positions = [(center_x, center_y - 1), (center_x, center_y), (center_x, center_y + 1)]
        else:  # Hàng dọc
            positions = [(center_x - 1, center_y), (center_x, center_y), (center_x + 1, center_y)]
        
        current_player = Board.BLACK
        for x, y in positions:
            game.board.board[x, y] = current_player
            current_player = Board.WHITE if current_player == Board.BLACK else Board.BLACK
        
        # Cập nhật lại trạng thái trò chơi
        game.current_player = current_player
        return game
    
    def _create_defense_position(self) -> GomokuGame:
        """Tạo tình huống phòng thủ (đối thủ có 3 quân liên tiếp)"""
        game = GomokuGame(self.board_size)
        
        center_x, center_y = self.board_size // 2, self.board_size // 2
        
        # Đặt quân trắng thành một hàng 3 quân
        positions = [(center_x, center_y - 1), (center_x, center_y), (center_x, center_y + 1)]
        
        # Đặt quân trắng
        for x, y in positions:
            game.board.board[x, y] = Board.WHITE
        
        # Đặt lượt chơi là quân đen để phòng thủ
        game.current_player = Board.BLACK
        return game
