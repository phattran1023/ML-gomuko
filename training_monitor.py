import matplotlib.pyplot as plt
import pandas as pd
import os
import json
import time
from typing import Dict, List, Optional

class TrainingMonitor:
    """Giám sát và hiển thị thông tin về quá trình huấn luyện"""
    
    def __init__(self, log_dir: str = "training_logs"):
        self.log_dir = log_dir
        self.session_id = time.strftime("%Y%m%d-%H%M%S")
        self.log_file = os.path.join(log_dir, f"training_log_{self.session_id}.json")
        self.stats = {
            'episodes': [],
            'black_wins': [],
            'white_wins': [],
            'draws': [],
            'opponent_types': [],
            'curriculum_stages': [],
            'game_lengths': [],
            'epsilon_values': [],
            'loss_values': []
        }
        
        # Tạo thư mục log nếu chưa tồn tại
        os.makedirs(log_dir, exist_ok=True)
    
    def log_episode(self, episode: int, black_wins: int, white_wins: int, 
                   draws: int, opponent_type: str, curriculum_stage: int, 
                   game_length: float, epsilon: float, loss: Optional[float] = None) -> None:
        """
        Ghi lại thông tin về một tập huấn luyện
        
        Args:
            episode: Số tập hiện tại
            black_wins: Số trận thắng của quân đen
            white_wins: Số trận thắng của quân trắng
            draws: Số trận hòa
            opponent_type: Loại đối thủ ('self', 'pool', 'random')
            curriculum_stage: Giai đoạn curriculum learning (0-4)
            game_length: Độ dài trò chơi
            epsilon: Giá trị epsilon hiện tại
            loss: Giá trị loss mới nhất (nếu có)
        """
        self.stats['episodes'].append(episode)
        self.stats['black_wins'].append(black_wins)
        self.stats['white_wins'].append(white_wins)
        self.stats['draws'].append(draws)
        self.stats['opponent_types'].append(opponent_type)
        self.stats['curriculum_stages'].append(curriculum_stage)
        self.stats['game_lengths'].append(game_length)
        self.stats['epsilon_values'].append(epsilon)
        self.stats['loss_values'].append(loss)
        
        # Ghi log vào file
        with open(self.log_file, 'w') as f:
            json.dump(self.stats, f)
    
    def plot_opponent_distribution(self, save_path: Optional[str] = None) -> None:
        """
        Vẽ biểu đồ phân phối các loại đối thủ
        
        Args:
            save_path: Đường dẫn để lưu biểu đồ (nếu cần)
        """
        opponent_counts = pd.Series(self.stats['opponent_types']).value_counts()
        
        plt.figure(figsize=(10, 6))
        opponent_counts.plot(kind='bar', color=['blue', 'green', 'orange'])
        plt.title('Phân phối loại đối thủ trong quá trình huấn luyện')
        plt.xlabel('Loại đối thủ')
        plt.ylabel('Số lượng tập')
        plt.xticks(rotation=0)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def plot_curriculum_progress(self, save_path: Optional[str] = None) -> None:
        """
        Vẽ biểu đồ tiến trình curriculum learning
        
        Args:
            save_path: Đường dẫn để lưu biểu đồ (nếu cần)
        """
        episodes = self.stats['episodes']
        stages = self.stats['curriculum_stages']
        
        plt.figure(figsize=(12, 6))
        plt.plot(episodes, stages, 'b-', linewidth=2)
        plt.scatter(episodes, stages, c='red', s=30)
        
        plt.title('Tiến trình Curriculum Learning')
        plt.xlabel('Tập huấn luyện')
        plt.ylabel('Giai đoạn (0-4)')
        plt.yticks(range(5), ['Bàn trống', 'Ít quân (3-5)', 'Nhiều quân (6-10)', 'Tấn công', 'Phòng thủ'])
        plt.grid(True, linestyle='--', alpha=0.7)
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def plot_win_rates(self, window_size: int = 50, save_path: Optional[str] = None) -> None:
        """
        Vẽ biểu đồ tỷ lệ thắng theo thời gian với đường trung bình động
        
        Args:
            window_size: Kích thước cửa sổ cho đường trung bình động
            save_path: Đường dẫn để lưu biểu đồ (nếu cần)
        """
        if not self.stats['episodes']:
            print("Không có dữ liệu để vẽ biểu đồ")
            return
            
        episodes = self.stats['episodes']
        total_games = [b + w + d for b, w, d in zip(
            self.stats['black_wins'], 
            self.stats['white_wins'], 
            self.stats['draws']
        )]
        
        black_rates = [b/max(1, t) for b, t in zip(self.stats['black_wins'], total_games)]
        white_rates = [w/max(1, t) for w, t in zip(self.stats['white_wins'], total_games)]
        draw_rates = [d/max(1, t) for d, t in zip(self.stats['draws'], total_games)]
        
        # Tính đường trung bình động
        window = min(window_size, len(episodes))
        if window > 0:
            black_moving_avg = pd.Series(black_rates).rolling(window=window).mean()
            white_moving_avg = pd.Series(white_rates).rolling(window=window).mean()
            
            plt.figure(figsize=(14, 7))
            plt.plot(episodes, black_rates, 'b-', alpha=0.3, label='Tỷ lệ thắng Đen')
            plt.plot(episodes, white_rates, 'r-', alpha=0.3, label='Tỷ lệ thắng Trắng')
            plt.plot(episodes, black_moving_avg, 'b-', linewidth=2, label=f'TB động Đen (win={window})')
            plt.plot(episodes, white_moving_avg, 'r-', linewidth=2, label=f'TB động Trắng (win={window})')
            
            plt.axhline(y=0.5, color='k', linestyle='--', alpha=0.5)
            plt.title('Tỷ lệ thắng theo thời gian')
            plt.xlabel('Tập huấn luyện')
            plt.ylabel('Tỷ lệ thắng')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            if save_path:
                plt.savefig(save_path)
            else:
                plt.show()
