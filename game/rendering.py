import pygame
import numpy as np
from .board import Board

class Renderer:
    """
    Hiển thị trực quan cho trò chơi Gomoku
    """
    
    def __init__(self, board_size: int = 15, cell_size: int = 40):
        """
        Khởi tạo renderer
        
        Args:
            board_size: Kích thước bàn cờ
            cell_size: Kích thước mỗi ô (pixel)
        """
        self.board_size = board_size
        self.cell_size = cell_size
        self.margin = cell_size
        
        # Màu sắc
        self.BACKGROUND = (240, 230, 200)  # Màu vàng nhạt cho bàn cờ
        self.GRID_COLOR = (150, 120, 90)   # Màu nâu cho lưới
        self.BLACK_COLOR = (0, 0, 0)       # Màu đen cho X
        self.WHITE_COLOR = (255, 0, 0)     # Màu đỏ cho O
        self.HIGHLIGHT_COLOR = (0, 200, 0) # Màu xanh lá cho nước đi mới nhất
        
        # Tính toán kích thước thực của bàn cờ 
        self.width = board_size * cell_size + 2 * self.margin
        self.height = board_size * cell_size + 2 * self.margin
        
        # Khởi tạo pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Gomoku (Caro) - X và O")
        self.font = pygame.font.Font(None, 36)
        # Font lớn hơn cho X và O
        self.piece_font = pygame.font.Font(None, int(cell_size * 1.5))
        
    def render_board(self, board_state: np.ndarray, last_move: tuple = None) -> None:
        """
        Hiển thị bàn cờ
        
        Args:
            board_state: Mảng 2D biểu diễn trạng thái bàn cờ
            last_move: Tọa độ (x, y) của nước đi cuối cùng
        """
        self.screen.fill(self.BACKGROUND)
        
        # Vẽ các ô vuông của lưới
        for i in range(self.board_size):
            for j in range(self.board_size):
                rect = pygame.Rect(
                    self.margin + j * self.cell_size,
                    self.margin + i * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                pygame.draw.rect(self.screen, self.GRID_COLOR, rect, 1)  # Vẽ viền của ô
                
                # Đánh dấu nước đi cuối cùng
                if last_move is not None and last_move == (i, j):
                    highlight_rect = pygame.Rect(
                        self.margin + j * self.cell_size + 2,  # +2 để tránh đè lên viền
                        self.margin + i * self.cell_size + 2,
                        self.cell_size - 4,
                        self.cell_size - 4
                    )
                    pygame.draw.rect(self.screen, self.HIGHLIGHT_COLOR, highlight_rect, 2)
        
        # Vẽ quân cờ (X và O)
        for i in range(self.board_size):
            for j in range(self.board_size):
                if board_state[i, j] != Board.EMPTY:
                    # Tính toán vị trí trung tâm của ô
                    center_x = self.margin + j * self.cell_size + self.cell_size // 2
                    center_y = self.margin + i * self.cell_size + self.cell_size // 2
                    
                    # Vẽ X hoặc O
                    if board_state[i, j] == Board.BLACK:
                        # Vẽ X
                        text = self.piece_font.render("X", True, self.BLACK_COLOR)
                    else:  # WHITE
                        # Vẽ O
                        text = self.piece_font.render("O", True, self.WHITE_COLOR)
                        
                    # Đặt text vào giữa ô
                    text_rect = text.get_rect(center=(center_x, center_y))
                    self.screen.blit(text, text_rect)
        
        pygame.display.flip()
        
    def render_game_over(self, winner: int) -> None:
        """
        Hiển thị thông báo kết thúc trò chơi
        
        Args:
            winner: Người chiến thắng (BLACK, WHITE hoặc None cho trận hòa)
        """
        if winner == Board.BLACK:
            message = "X thắng!"
            color = self.BLACK_COLOR
        elif winner == Board.WHITE:
            message = "O thắng!"
            color = self.WHITE_COLOR
        else:
            message = "Trận hòa!"
            color = (100, 100, 100)
            
        # Tạo nền mờ cho thông báo
        overlay = pygame.Surface((self.width, self.height))
        overlay.set_alpha(180)  # Độ mờ
        overlay.fill((220, 220, 220))  # Màu xám nhạt
        self.screen.blit(overlay, (0, 0))
        
        # Vẽ thông báo
        text = self.font.render(message, True, color)
        text_rect = text.get_rect(center=(self.width // 2, self.height // 2))
        self.screen.blit(text, text_rect)
        pygame.display.flip()
        
    def close(self) -> None:
        """Đóng cửa sổ hiển thị"""
        pygame.quit()
