import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Tuple, Dict, Any, List
from .base_agent import Agent
from game.board import Board

class DQN(nn.Module):
    """
    Mô hình mạng neural cho Deep Q-Network với kiến trúc cải tiến
    """
    def __init__(self, board_size: int, hidden_size: int = 512):
        super(DQN, self).__init__()
        
        # Input: 3 kênh (Quân đen, quân trắng, lượt hiện tại)
        self.board_size = board_size
        input_channels = 3
        
        # Sử dụng mạng CNN sâu hơn với batch normalization
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        # Residual connections
        self.residual1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256)
        )
        
        self.residual2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256)
        )
        
        # Policy head (output: hành động)
        self.policy_conv = nn.Conv2d(256, 128, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(128)
        self.policy_fc = nn.Linear(128 * board_size * board_size, board_size * board_size)
        
        # Value head (output: ước lượng giá trị trạng thái)
        self.value_conv = nn.Conv2d(256, 128, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(128)
        self.value_fc1 = nn.Linear(128 * board_size * board_size, hidden_size)
        self.value_fc2 = nn.Linear(hidden_size, 1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x có kích thước [batch_size, 3, board_size, board_size]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Residual connections
        res = x
        x = self.residual1(x)
        x = x + res
        x = F.relu(x)
        
        res = x
        x = self.residual2(x)
        x = x + res
        x = F.relu(x)
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, 128 * self.board_size * self.board_size)
        policy = self.policy_fc(policy)
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, 128 * self.board_size * self.board_size)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))  # Đầu ra trong khoảng [-1, 1]
        
        return policy, value

class ReplayBuffer:
    """
    Bộ nhớ ưu tiên để lưu và lấy mẫu các trải nghiệm
    """
    def __init__(self, capacity: int, alpha: float = 0.6, beta_start: float = 0.4, beta_end: float = 1.0, beta_frames: int = 100000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha  # Độ ưu tiên
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_frames = beta_frames
        self.frame = 1
        
    def push(self, state: np.ndarray, action: Tuple[int, int], reward: float, 
             next_state: np.ndarray, done: bool) -> None:
        """
        Thêm một trải nghiệm vào bộ nhớ
        """
        max_prio = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.priorities[self.position] = max_prio
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size: int) -> Tuple:
        """
        Lấy mẫu ưu tiên từ bộ nhớ
        """
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.position]
            
        # Tính beta hiện tại
        beta = self.beta_start + (self.beta_end - self.beta_start) * min(1.0, self.frame / self.beta_frames)
        self.frame += 1
        
        # Chuyển đổi priorities thành probabilities
        probs = prios ** self.alpha
        probs /= probs.sum()
        
        # Lấy mẫu theo probabilities
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        # Tính weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.FloatTensor(weights)
        
        batch = list(zip(*samples))
        states = np.array(batch[0])
        actions = batch[1]
        rewards = batch[2]
        next_states = np.array(batch[3])
        dones = batch[4]
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices, priorities):
        """
        Cập nhật priorities sau khi huấn luyện
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
    
    def __len__(self) -> int:
        return len(self.buffer)

class RLAgent(Agent):
    """
    Agent sử dụng Deep Q-Learning cải tiến với PER và Dueling DQN
    """
    
    def __init__(self, player_id: int, board_size: int = 15, 
                 learning_rate: float = 0.0001, gamma: float = 0.99,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.1,
                 epsilon_decay: float = 0.9999, memory_size: int = 100000,
                 batch_size: int = 128, target_update: int = 1000,
                 double_dqn: bool = True):
        """
        Khởi tạo RL Agent
        
        Args:
            player_id: Định danh người chơi (BLACK hoặc WHITE)
            board_size: Kích thước bàn cờ
            learning_rate: Tốc độ học
            gamma: Hệ số giảm giá
            epsilon_start: Giá trị epsilon ban đầu cho chính sách epsilon-greedy
            epsilon_end: Giá trị epsilon cuối cùng
            epsilon_decay: Tốc độ giảm epsilon
            memory_size: Kích thước bộ nhớ
            batch_size: Kích thước batch khi huấn luyện
            target_update: Số bước để cập nhật mạng target
            double_dqn: Sử dụng Double DQN hay không
        """
        super().__init__(player_id)
        
        # Kiểm tra CUDA và thiết lập device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
        
        self.board_size = board_size
        
        # Siêu tham số
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.double_dqn = double_dqn
        
        # Mạng neural và bộ nhớ
        self.policy_net = DQN(board_size).to(self.device)
        self.target_net = DQN(board_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(memory_size)
        
        self.steps_done = 0
        self.training_info = {
            'losses': [],
            'q_values': [],
            'rewards': [],
            'epsilons': []
        }
        
        # Biến debug
        self.debug_top_moves = []
        self.debug_top_values = []
    
    def _preprocess_state(self, game_state: Dict[str, Any]) -> torch.Tensor:
        """
        Chuyển đổi trạng thái trò chơi thành tensor đầu vào cho mạng neural
        
        Args:
            game_state: Trạng thái trò chơi
            
        Returns:
            torch.Tensor: Tensor biểu diễn trạng thái
        """
        board = game_state['board']
        current_player = game_state['current_player']
        
        # 3 kênh: quân đen, quân trắng, lượt hiện tại
        state = np.zeros((3, self.board_size, self.board_size), dtype=np.float32)
        
        # Kênh 0: vị trí quân đen
        state[0] = (board == Board.BLACK).astype(np.float32)
        
        # Kênh 1: vị trí quân trắng
        state[1] = (board == Board.WHITE).astype(np.float32)
        
        # Kênh 2: lượt hiện tại
        state[2] = np.ones((self.board_size, self.board_size), dtype=np.float32) * (current_player == self.player_id)
        
        return torch.FloatTensor(state).unsqueeze(0).to(self.device)
    
    def get_action(self, game_state: Dict[str, Any]) -> Tuple[int, int]:
        """
        Chọn một hành động dựa trên chính sách epsilon-greedy
        
        Args:
            game_state: Trạng thái trò chơi hiện tại
            
        Returns:
            Tuple[int, int]: Tọa độ (x, y) của nước đi được chọn
        """
        valid_moves = game_state['valid_moves']
        
        if not valid_moves:
            raise ValueError("Không có nước đi hợp lệ")
        
        # Xóa thông tin debug cũ
        self.debug_top_moves = []
        self.debug_top_values = []
        
        # Thăm dò ngẫu nhiên
        if random.random() < self.epsilon:
            chosen_move = random.choice(valid_moves)
            print(f"[AI quyết định] Đi ngẫu nhiên: {chosen_move} (epsilon={self.epsilon:.4f})")
            return chosen_move
        
        # Chọn nước đi tốt nhất theo mô hình
        with torch.no_grad():
            state = self._preprocess_state(game_state)
            q_values, _ = self.policy_net(state)
            q_values = q_values.squeeze(0)
            
            # Mặt nạ cho các nước đi hợp lệ
            valid_mask = torch.zeros(self.board_size * self.board_size, device=self.device)
            for x, y in valid_moves:
                idx = x * self.board_size + y
                valid_mask[idx] = 1
            
            # Chỉ xét Q-value cho các nước đi hợp lệ
            masked_q_values = q_values * valid_mask
            masked_q_values[valid_mask == 0] = float('-inf')
            
            # Lưu top 3 nước đi và giá trị Q cho debug
            if len(valid_moves) >= 3:
                top_k = min(3, len(valid_moves))
                top_indices = torch.topk(masked_q_values, top_k).indices.cpu().numpy()
                top_values = torch.topk(masked_q_values, top_k).values.cpu().numpy()
                
                # Chuyển đổi chỉ số thành tọa độ (x, y)
                self.debug_top_moves = [(idx // self.board_size, idx % self.board_size) for idx in top_indices]
                self.debug_top_values = [float(val) for val in top_values]
                
                # Hiển thị thông tin về quyết định của AI
                print("\n[PHÂN TÍCH QUYẾT ĐỊNH CỦA AI]")
                for i, ((x, y), val) in enumerate(zip(self.debug_top_moves, self.debug_top_values)):
                    print(f"  Lựa chọn #{i+1}: ({x}, {y}) với Q-value = {val:.6f}")
            
            # Chọn nước đi có Q-value cao nhất
            best_idx = torch.argmax(masked_q_values).item()
            x = best_idx // self.board_size
            y = best_idx % self.board_size
            
            self.training_info['q_values'].append(masked_q_values.max().item())
            
            return (x, y)
    
    def update(self, game_state: Dict[str, Any], action: Tuple[int, int], reward: float, 
               next_state: Dict[str, Any], done: bool) -> None:
        """
        Cập nhật mô hình sau mỗi hành động
        
        Args:
            game_state: Trạng thái trò chơi trước khi thực hiện hành động
            action: Hành động đã thực hiện (x, y)
            reward: Phần thưởng nhận được
            next_state: Trạng thái trò chơi sau khi thực hiện hành động
            done: True nếu trò chơi kết thúc, False nếu chưa
        """
        # Thêm trải nghiệm vào bộ nhớ
        state_tensor = self._preprocess_state(game_state).cpu().numpy()[0]
        next_state_tensor = self._preprocess_state(next_state).cpu().numpy()[0]
        
        self.memory.push(state_tensor, action, reward, next_state_tensor, done)
        self.training_info['rewards'].append(reward)
        
        # Huấn luyện mô hình nếu có đủ dữ liệu
        if len(self.memory) < self.batch_size:
            return
        
        loss = self._train_model()
        if loss is not None:
            self.training_info['losses'].append(loss)
        
        # Cập nhật epsilon (giảm dần)
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.training_info['epsilons'].append(self.epsilon)
        
        # Cập nhật mạng target
        self.steps_done += 1
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def _train_model(self) -> float:
        """
        Huấn luyện mô hình DQN
        
        Returns:
            float: Giá trị loss trong batch này
        """
        # Lấy batch từ bộ nhớ
        state_batch, action_batch, reward_batch, next_state_batch, done_batch, indices, weights = self.memory.sample(self.batch_size)
        
        # Chuyển đổi thành tensor
        state_batch = torch.FloatTensor(np.array(state_batch)).to(self.device)
        action_batch = torch.LongTensor([(x * self.board_size + y) for x, y in action_batch]).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(next_state_batch)).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device)
        weights = weights.to(self.device)
        
        # Tính Q-values hiện tại
        q_values, _ = self.policy_net(state_batch)
        state_action_values = q_values.gather(1, action_batch)
        
        # Tính Q-values mục tiêu với Double DQN
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: Sử dụng policy network để chọn hành động, target network để đánh giá giá trị
                next_q_values, _ = self.policy_net(next_state_batch)
                next_actions = next_q_values.max(1)[1].unsqueeze(1)
                next_q_target, _ = self.target_net(next_state_batch)
                next_state_values = next_q_target.gather(1, next_actions).squeeze(1)
            else:
                # Vanilla DQN
                next_q_values, _ = self.target_net(next_state_batch)
                next_state_values = next_q_values.max(1)[0]
            
            # Tính target
            expected_state_action_values = reward_batch + (1 - done_batch) * self.gamma * next_state_values
        
        # Tính mất mát với Huber Loss (ít nhạy cảm hơn với outliers)
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1), reduction='none')
        
        # Áp dụng weights từ prioritized experience replay
        weighted_loss = (loss * weights.unsqueeze(1)).mean()
        
        # Cập nhật ưu tiên trong replay buffer
        with torch.no_grad():
            priorities = loss.detach().cpu().numpy() + 1e-5  # Tránh priority = 0
            self.memory.update_priorities(indices, priorities)
        
        # Backpropagation
        self.optimizer.zero_grad()
        weighted_loss.backward()
        
        # Gradient clipping để tránh exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        
        self.optimizer.step()
        
        return weighted_loss.item()
    
    def save(self, filepath: str) -> None:
        """
        Lưu mô hình
        
        Args:
            filepath: Đường dẫn tệp để lưu mô hình
        """
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'training_info': self.training_info
        }, filepath)
        
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Tải mô hình
        
        Args:
            filepath: Đường dẫn tệp để tải mô hình
        """
        if not os.path.exists(filepath):
            print(f"Model file not found: {filepath}")
            return
        
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            
            # Đảm bảo các trường quan trọng được tải chính xác
            if 'epsilon' in checkpoint:
                self.epsilon = checkpoint['epsilon']
            else:
                self.epsilon = self.epsilon_end  # Sử dụng giá trị an toàn
                print("Cảnh báo: epsilon không có trong checkpoint, sử dụng epsilon_end")
                
            self.steps_done = checkpoint.get('steps_done', 0)
            
            if 'training_info' in checkpoint:
                self.training_info = checkpoint['training_info']
                
            print(f"Model loaded from {filepath}")
            print(f"Current epsilon: {self.epsilon:.4f}, Steps done: {self.steps_done}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Thử tải mô hình với cấu hình thay thế...")
            try:
                checkpoint = torch.load(filepath, map_location=self.device)
                self.policy_net.load_state_dict(checkpoint['policy_net'])
                self.target_net.load_state_dict(checkpoint['policy_net'])  # Backup: sử dụng policy net cho target net
                self.epsilon = self.epsilon_end  # Sử dụng giá trị thấp để khai thác hơn là thăm dò
                print(f"Đã tải mô hình thành công với cấu hình thay thế. Epsilon = {self.epsilon}")
            except Exception as e2:
                print(f"Tải mô hình hoàn toàn thất bại: {e2}")
    
    def get_training_stats(self) -> Dict[str, List[float]]:
        """
        Lấy thống kê huấn luyện
        
        Returns:
            Dict[str, List[float]]: Thông tin huấn luyện
        """
        return self.training_info
