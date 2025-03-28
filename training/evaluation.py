import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List, Tuple, Any
from game.board import Board
from game.game import GomokuGame
from agents.base_agent import Agent

class Evaluator:
    """
    Lớp đánh giá hiệu suất của các agent
    """
    
    def __init__(self, config_path: str = "config.json"):
        """
        Khởi tạo evaluator
        
        Args:
            config_path: Đường dẫn đến tệp cấu hình
        """
        # Tải cấu hình
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.board_size = self.config['game']['board_size']
        self.win_length = self.config['game']['win_length']
        self.num_games = self.config['evaluation']['num_games']
    
    def evaluate(self, agent1: Agent, agent2: Agent, num_games: int = None, 
                verbose: bool = True) -> Dict[str, float]:
        """
        Đánh giá hiệu suất của agent1 so với agent2
        
        Args:
            agent1: Agent cần đánh giá
            agent2: Agent đối thủ
            num_games: Số trận đấu (mặc định sử dụng từ cấu hình)
            verbose: Hiển thị thông tin đánh giá
            
        Returns:
            Dict[str, float]: Kết quả đánh giá
        """
        if num_games is None:
            num_games = self.num_games
        
        wins = 0
        draws = 0
        losses = 0
        total_moves = 0
        
        # Tạo progress bar nếu verbose
        games_iter = tqdm(range(num_games)) if verbose else range(num_games)
        
        for game_idx in games_iter:
            # Đảo lượt chơi sau mỗi trận
            if game_idx % 2 == 0:
                black_agent, white_agent = agent1, agent2
            else:
                black_agent, white_agent = agent2, agent1
            
            # Khởi tạo trò chơi mới
            game = GomokuGame(self.board_size, self.win_length)
            
            # Đặt lại agent
            black_agent.reset()
            white_agent.reset()
            
            # Chơi trò chơi cho đến khi kết thúc
            moves = 0
            while not game.is_game_over():
                current_state = game.get_state()
                current_player = current_state['current_player']
                
                # Xác định agent hiện tại
                current_agent = black_agent if current_player == Board.BLACK else white_agent
                
                # Lấy hành động từ agent
                action = current_agent.get_action(current_state)
                
                # Thực hiện hành động
                game.make_move(action[0], action[1])
                
                moves += 1
            
            # Cập nhật thống kê
            total_moves += moves
            
            # Xác định kết quả
            winner = game.get_winner()
            
            if winner is None:
                draws += 1
            elif (game_idx % 2 == 0 and winner == Board.BLACK) or (game_idx % 2 == 1 and winner == Board.WHITE):
                # Agent1 thắng
                wins += 1
            else:
                # Agent1 thua
                losses += 1
            
            # Hiển thị thông tin
            if verbose and (game_idx + 1) % 10 == 0:
                win_rate = wins / (game_idx + 1)
                draw_rate = draws / (game_idx + 1)
                avg_moves = total_moves / (game_idx + 1)
                tqdm.write(f"Game: {game_idx+1}/{num_games}, Win Rate: {win_rate:.4f}, Draw Rate: {draw_rate:.4f}, Avg Moves: {avg_moves:.1f}")
        
        # Tính thống kê cuối cùng
        win_rate = wins / num_games
        draw_rate = draws / num_games
        loss_rate = losses / num_games
        avg_moves = total_moves / num_games
        
        results = {
            'win_rate': win_rate,
            'draw_rate': draw_rate,
            'loss_rate': loss_rate,
            'avg_moves': avg_moves,
            'total_games': num_games
        }
        
        # Hiển thị kết quả
        if verbose:
            print("\nEvaluation Results:")
            print(f"Win Rate: {win_rate:.4f}")
            print(f"Draw Rate: {draw_rate:.4f}")
            print(f"Loss Rate: {loss_rate:.4f}")
            print(f"Average Moves: {avg_moves:.1f}")
            print(f"Total Games: {num_games}")
        
        return results
    
    def plot_training_history(self, history: Dict[str, List[float]], save_path: str = None) -> None:
        """
        Vẽ biểu đồ lịch sử huấn luyện
        
        Args:
            history: Lịch sử huấn luyện
            save_path: Đường dẫn để lưu biểu đồ (nếu None sẽ hiển thị biểu đồ)
        """
        plt.figure(figsize=(15, 10))
        
        # Vẽ tỷ lệ thắng
        plt.subplot(3, 1, 1)
        plt.plot(history['win_rate'])
        plt.title('Win Rate')
        plt.xlabel('Episode')
        plt.ylabel('Win Rate')
        plt.grid(True)
        
        # Vẽ phần thưởng
        plt.subplot(3, 1, 2)
        plt.plot(history['rewards'])
        plt.title('Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid(True)
        
        # Vẽ độ dài trận đấu trung bình
        plt.subplot(3, 1, 3)
        plt.plot(history['avg_game_length'])
        plt.title('Average Game Length')
        plt.xlabel('Episode')
        plt.ylabel('Moves')
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
