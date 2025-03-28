import os
import json
import time
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Dict, Any
from game.board import Board
from game.game import GomokuGame
from agents.base_agent import Agent

class Trainer:
    """
    Lớp huấn luyện AI
    """
    
    def __init__(self, config_path: str = "config.json"):
        """
        Khởi tạo trainer
        
        Args:
            config_path: Đường dẫn đến tệp cấu hình
        """
        # Tải cấu hình
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.board_size = self.config['game']['board_size']
        self.win_length = self.config['game']['win_length']
        self.episodes = self.config['training']['episodes']
        self.models_dir = self.config['models_dir']
        
        # Tạo thư mục lưu mô hình nếu chưa tồn tại
        os.makedirs(self.models_dir, exist_ok=True)
    
    def train(self, agent1: Agent, agent2: Agent = None, save_interval: int = 100, 
              verbose: bool = True) -> Dict[str, List[float]]:
        """
        Huấn luyện agent qua nhiều trận đấu
        
        Args:
            agent1: Agent cần huấn luyện
            agent2: Agent đối thủ (nếu None sẽ sử dụng bản sao của agent1)
            save_interval: Số trận đấu giữa mỗi lần lưu mô hình
            verbose: Hiển thị thông tin trong quá trình huấn luyện
            
        Returns:
            Dict[str, List[float]]: Lịch sử chỉ số huấn luyện
        """
        history = {
            'rewards': [],
            'win_rate': [],
            'avg_game_length': []
        }
        
        total_wins = 0
        total_games = 0
        total_moves = 0
        
        # Tạo progress bar nếu verbose
        episodes_iter = tqdm(range(self.episodes)) if verbose else range(self.episodes)
        
        for episode in episodes_iter:
            # Khởi tạo trò chơi mới
            game = GomokuGame(self.board_size, self.win_length)
            
            # Đặt lại agent
            agent1.reset()
            if agent2:
                agent2.reset()
            
            # Lưu lại trạng thái và hành động
            states = []
            actions = []
            
            # Đếm số lượt
            moves = 0
            
            # Chơi trò chơi cho đến khi kết thúc
            while not game.is_game_over():
                current_state = game.get_state()
                current_player = current_state['current_player']
                
                # Xác định agent hiện tại
                current_agent = agent1 if current_player == agent1.player_id else agent2
                
                # Lấy hành động từ agent
                action = current_agent.get_action(current_state)
                
                # Lưu trạng thái và hành động
                if current_agent == agent1:
                    states.append(current_state)
                    actions.append(action)
                
                # Thực hiện hành động
                game.make_move(action[0], action[1])
                
                moves += 1
            
            # Tính phần thưởng
            winner = game.get_winner()
            
            if winner == agent1.player_id:
                reward = 1.0  # Thắng
                total_wins += 1
            elif winner is None:
                reward = 0.0  # Hòa
            else:
                reward = -1.0  # Thua
            
            # Cập nhật thống kê
            total_games += 1
            total_moves += moves
            
            # Cập nhật agent1
            for i in range(len(states)):
                state = states[i]
                action = actions[i]
                next_state = game.get_state() if i == len(states) - 1 else states[i + 1]
                done = i == len(states) - 1
                
                agent1.update(state, action, reward, next_state, done)
            
            # Cập nhật lịch sử
            history['rewards'].append(reward)
            history['win_rate'].append(total_wins / total_games)
            history['avg_game_length'].append(total_moves / total_games)
            
            # Hiển thị thông tin
            if verbose and (episode + 1) % 10 == 0:
                win_rate = total_wins / total_games
                avg_moves = total_moves / total_games
                tqdm.write(f"Episode: {episode+1}/{self.episodes}, Win Rate: {win_rate:.4f}, Avg Moves: {avg_moves:.1f}")
            
            # Lưu mô hình
            if (episode + 1) % save_interval == 0:
                model_path = os.path.join(self.models_dir, f"model_episode_{episode+1}.pt")
                agent1.save(model_path)
                if verbose:
                    tqdm.write(f"Model saved to {model_path}")
        
        # Lưu mô hình cuối cùng
        final_model_path = os.path.join(self.models_dir, "model_final.pt")
        agent1.save(final_model_path)
        if verbose:
            print(f"Final model saved to {final_model_path}")
        
        return history
    
    def self_play(self, agent: Agent, save_interval: int = 100, verbose: bool = True) -> Dict[str, List[float]]:
        """
        Huấn luyện thông qua tự chơi
        
        Args:
            agent: Agent cần huấn luyện
            save_interval: Số trận đấu giữa mỗi lần lưu mô hình
            verbose: Hiển thị thông tin trong quá trình huấn luyện
            
        Returns:
            Dict[str, List[float]]: Lịch sử chỉ số huấn luyện
        """
        # Tạo bản sao của agent
        agent_copy = type(agent)(agent.player_id)
        
        # Thiết lập agent bản sao sử dụng cùng tham số
        if hasattr(agent, 'policy_net') and hasattr(agent_copy, 'policy_net'):
            agent_copy.policy_net.load_state_dict(agent.policy_net.state_dict())
            agent_copy.target_net.load_state_dict(agent.target_net.state_dict())
        
        # Thiết lập agent bản sao là đối thủ
        agent_copy.player_id = Board.WHITE if agent.player_id == Board.BLACK else Board.BLACK
        
        return self.train(agent, agent_copy, save_interval, verbose)
