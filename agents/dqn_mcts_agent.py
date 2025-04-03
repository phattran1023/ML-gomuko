import os
import math
import random
import numpy as np
import torch
from typing import Tuple, Dict, List, Any, Optional
from .base_agent import Agent
from .rl_agent import RLAgent, DQN
from game.board import Board

class MCTSNode:
    """Node trong cây tìm kiếm MCTS"""
    def __init__(self, state=None, parent=None, action=None):
        self.state = state  # Trạng thái trò chơi
        self.parent = parent  # Node cha
        self.action = action  # Hành động dẫn đến node này
        
        self.children = {}  # action -> MCTSNode
        self.visit_count = 0  # Số lần thăm
        self.value_sum = 0  # Tổng giá trị
        self.mean_value = 0  # Giá trị trung bình
        
        self.prior = 0  # Xác suất prior từ mạng neural
        
    def is_expanded(self) -> bool:
        """Kiểm tra node đã được mở rộng chưa"""
        return len(self.children) > 0
    
    def select_child(self, c_puct: float = 1.0) -> Tuple[Tuple[int, int], Any]:
        """
        Chọn node con sử dụng PUCT (Polynomial Upper Confidence Trees)
        
        Args:
            c_puct: Hằng số kiểm soát mức độ thăm dò
            
        Returns:
            Tuple[action, child_node]: Hành động được chọn và node con tương ứng
        """
        # Công thức PUCT: Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        best_score = -float('inf')
        best_action = None
        best_child = None
        
        # Tổng số lần thăm của node cha
        parent_visit_count = self.visit_count
        
        for action, child in self.children.items():
            # Exploitation: Giá trị trung bình của node con
            q_value = child.mean_value
            
            # Exploration: Khuyến khích thăm dò các node ít được thăm nhưng có prior cao
            exploration_term = c_puct * child.prior * (math.sqrt(parent_visit_count) / (1 + child.visit_count))
            
            # Tổng điểm
            score = q_value + exploration_term
            
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        
        return best_action, best_child
    
    def expand(self, state: Dict[str, Any], action_priors: Dict[Tuple[int, int], float]) -> None:
        """
        Mở rộng node với các node con và prior
        
        Args:
            state: Trạng thái trò chơi
            action_priors: Dictionary ánh xạ từ hành động đến prior
        """
        for action, prior in action_priors.items():
            if action not in self.children:
                # Tạo node con mới
                child_node = MCTSNode(parent=self, action=action)
                child_node.prior = prior
                self.children[action] = child_node
    
    def update(self, value: float) -> None:
        """
        Cập nhật giá trị của node sau khi simulation
        
        Args:
            value: Giá trị cần cập nhật
        """
        self.visit_count += 1
        self.value_sum += value
        self.mean_value = self.value_sum / self.visit_count

class DQNMCTS:
    """
    Monte Carlo Tree Search kết hợp với DQN
    """
    def __init__(self, dqn_model: DQN, player_id: int, board_size: int = 15, 
                num_simulations: int = 100, c_puct: float = 1.0,
                temperature: float = 1.0, dirichlet_noise: bool = False,
                device = None):
        """
        Khởi tạo MCTS với DQN
        
        Args:
            dqn_model: Mô hình DQN được huấn luyện
            player_id: ID của người chơi (BLACK hoặc WHITE)
            board_size: Kích thước bàn cờ
            num_simulations: Số lần mô phỏng cho mỗi nước đi
            c_puct: Hằng số PUCT cho exploration
            temperature: Tham số nhiệt độ cho việc chọn nước đi
            dirichlet_noise: Sử dụng nhiễu Dirichlet trong quá trình huấn luyện
            device: Thiết bị tính toán (cuda hoặc cpu)
        """
        self.dqn_model = dqn_model
        self.player_id = player_id
        self.board_size = board_size
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.dirichlet_noise = dirichlet_noise
        self.device = device
    
    def _evaluate_state(self, state: Dict[str, Any]) -> Tuple[Dict[Tuple[int, int], float], float]:
        """
        Đánh giá trạng thái bằng mô hình DQN
        
        Args:
            state: Trạng thái trò chơi
            
        Returns:
            Tuple[action_priors, value]: Các prior hành động và giá trị trạng thái
        """
        # Chuẩn bị input
        state_tensor = self._prepare_state_tensor(state)
        
        # Đánh giá bằng DQN
        with torch.no_grad():
            q_values, state_value = self.dqn_model(state_tensor)
            
        q_values = q_values.squeeze(0).cpu().numpy()
        state_value = state_value.item()
        
        # Convert q_values thành priors cho các hành động hợp lệ
        valid_moves = state['valid_moves']
        valid_indices = [(x * self.board_size + y) for x, y in valid_moves]
        
        # Lấy q-values cho các nước đi hợp lệ
        valid_q_values = q_values[valid_indices]
        
        # Chuyển q-values thành phân phối xác suất bằng softmax
        probs = self._softmax(valid_q_values / self.temperature)
        
        # Ánh xạ probs với các action tương ứng
        action_priors = {action: prob for action, prob in zip(valid_moves, probs)}
        
        return action_priors, state_value
    
    def _prepare_state_tensor(self, state: Dict[str, Any]) -> torch.Tensor:
        """
        Chuẩn bị tensor đầu vào từ trạng thái trò chơi
        
        Args:
            state: Trạng thái trò chơi
            
        Returns:
            torch.Tensor: Tensor đầu vào cho mô hình DQN
        """
        board = state['board']
        current_player = state['current_player']
        
        # 3 kênh: quân đen, quân trắng, lượt hiện tại
        input_tensor = np.zeros((3, self.board_size, self.board_size), dtype=np.float32)
        
        # Kênh 0: vị trí quân đen
        input_tensor[0] = (board == Board.BLACK).astype(np.float32)
        
        # Kênh 1: vị trí quân trắng
        input_tensor[1] = (board == Board.WHITE).astype(np.float32)
        
        # Kênh 2: lượt hiện tại
        input_tensor[2] = np.ones((self.board_size, self.board_size), dtype=np.float32) * (current_player == self.player_id)
        
        return torch.FloatTensor(input_tensor).unsqueeze(0).to(self.device)
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """
        Áp dụng hàm softmax cho mảng
        
        Args:
            x: Mảng đầu vào
            
        Returns:
            np.ndarray: Xác suất sau khi áp dụng softmax
        """
        exp_x = np.exp(x - np.max(x))  # Trừ max để tránh overflow
        return exp_x / exp_x.sum()
    
    def _simulate_move(self, state: Dict[str, Any], action: Tuple[int, int]) -> Dict[str, Any]:
        """
        Mô phỏng một nước đi và trả về trạng thái mới
        
        Args:
            state: Trạng thái hiện tại
            action: Hành động cần mô phỏng
            
        Returns:
            Dict[str, Any]: Trạng thái mới sau khi thực hiện hành động
        """
        row, col = action
        new_board = state['board'].copy()
        new_board[row, col] = state['current_player']
        
        # Chuyển người chơi
        next_player = Board.BLACK if state['current_player'] == Board.WHITE else Board.WHITE
        
        # Cập nhật danh sách nước đi hợp lệ
        valid_moves = [(r, c) for r, c in state['valid_moves'] if not (r == row and c == col)]
        
        return {
            'board': new_board,
            'current_player': next_player,
            'valid_moves': valid_moves
        }
    
    def _check_game_over(self, state: Dict[str, Any]) -> Tuple[bool, Optional[int]]:
        """
        Kiểm tra xem trò chơi đã kết thúc chưa
        
        Args:
            state: Trạng thái trò chơi
            
        Returns:
            Tuple[is_over, winner]: True nếu trò chơi kết thúc và người chiến thắng (nếu có)
        """
        # Kiểm tra hòa: không còn nước đi hợp lệ
        if not state['valid_moves']:
            return True, 0
        
        board = state['board']
        win_len = 5  # Số quân liên tiếp để thắng
        
        # Kiểm tra hàng ngang, dọc và chéo
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # Ngang, dọc, chéo xuống, chéo lên
        
        for r in range(self.board_size):
            for c in range(self.board_size):
                if board[r, c] == Board.EMPTY:
                    continue
                    
                player = board[r, c]
                
                for dr, dc in directions:
                    count = 0
                    for i in range(win_len):
                        nr, nc = r + i * dr, c + i * dc
                        if 0 <= nr < self.board_size and 0 <= nc < self.board_size and board[nr, nc] == player:
                            count += 1
                        else:
                            break
                    
                    if count == win_len:
                        return True, player
        
        # Trò chơi chưa kết thúc
        return False, None
    
    def search(self, state: Dict[str, Any]) -> Dict[Tuple[int, int], float]:
        """
        Thực hiện tìm kiếm MCTS và trả về phân phối xác suất cho các nước đi
        
        Args:
            state: Trạng thái trò chơi hiện tại
            
        Returns:
            Dict[Tuple[int, int], float]: Ánh xạ từ hành động đến xác suất
        """
        # Khởi tạo node gốc
        root = MCTSNode(state=state)
        
        # Đánh giá và mở rộng node gốc
        action_priors, _ = self._evaluate_state(state)
        
        # Thêm nhiễu Dirichlet để tăng tính khám phá (nếu cần)
        if self.dirichlet_noise:
            dirichlet_alpha = 0.3  # Tham số cho nhiễu Dirichlet
            noise = np.random.dirichlet([dirichlet_alpha] * len(action_priors))
            
            # Trộn prior với nhiễu (ví dụ: 75% prior + 25% nhiễu)
            noise_weight = 0.25
            action_priors = {
                action: (1 - noise_weight) * prior + noise_weight * noise[i]
                for i, (action, prior) in enumerate(action_priors.items())
            }
        
        # Mở rộng node gốc
        root.expand(state, action_priors)
        
        # Thực hiện các simulation
        for _ in range(self.num_simulations):
            # Chọn: đi từ node gốc đến node lá
            node = root
            search_path = [node]
            current_state = state.copy()
            
            # Đi xuống cây cho đến khi gặp node lá (không mở rộng hoặc node kết thúc)
            while node.is_expanded():
                action, node = node.select_child(self.c_puct)
                current_state = self._simulate_move(current_state, action)
                search_path.append(node)
                
                # Kiểm tra kết thúc trò chơi
                game_over, winner = self._check_game_over(current_state)
                if game_over:
                    break
            
            # Đánh giá node lá
            if not game_over:
                # Mở rộng node lá
                action_priors, leaf_value = self._evaluate_state(current_state)
                node.expand(current_state, action_priors)
            else:
                # Trò chơi kết thúc, xác định giá trị dựa trên người thắng
                if winner == 0:  # Hòa
                    leaf_value = 0.0
                else:
                    # +1 nếu thắng, -1 nếu thua (từ góc nhìn của người chơi hiện tại)
                    leaf_value = 1.0 if winner == self.player_id else -1.0
            
            # Cập nhật: cập nhật giá trị cho tất cả các node trong đường đi
            # Giá trị được truyền ngược lên cây từ góc nhìn của người chơi tương ứng
            for node in reversed(search_path):
                # Đổi dấu giá trị khi chuyển lượt
                current_player = node.state['current_player'] if node.state else self.player_id
                node_value = leaf_value if current_player == self.player_id else -leaf_value
                node.update(node_value)
        
        # Tính xác suất dựa trên số lần thăm mỗi node con
        visit_counts = {}
        for action, child in root.children.items():
            visit_counts[action] = child.visit_count
        
        # Chuyển đổi số lần thăm thành xác suất
        total_visits = sum(visit_counts.values())
        
        if self.temperature == 0:  # Deterministic - luôn chọn nước đi tốt nhất
            best_action = max(visit_counts.items(), key=lambda x: x[1])[0]
            probs = {action: 1.0 if action == best_action else 0.0 for action in visit_counts}
        else:
            # Áp dụng temperature
            scaled_visits = {action: count ** (1.0 / self.temperature) for action, count in visit_counts.items()}
            total_scaled = sum(scaled_visits.values())
            probs = {action: visits / total_scaled for action, visits in scaled_visits.items()}
        
        return probs

class DQNMCTSAgent(Agent):
    """
    Agent kết hợp DQN và MCTS
    """
    def __init__(self, player_id: int, dqn_agent: RLAgent = None, model_path: str = None,
                 board_size: int = 15, num_simulations: int = 100, temperature: float = 0.1):
        """
        Khởi tạo agent kết hợp DQN và MCTS
        
        Args:
            player_id: ID của người chơi (BLACK hoặc WHITE)
            dqn_agent: Agent DQN đã được huấn luyện (nếu có)
            model_path: Đường dẫn đến mô hình (nếu không cung cấp dqn_agent)
            board_size: Kích thước bàn cờ
            num_simulations: Số lần mô phỏng MCTS cho mỗi nước đi
            temperature: Tham số nhiệt độ cho việc chọn nước đi
        """
        super().__init__(player_id)
        
        self.board_size = board_size
        self.num_simulations = num_simulations
        self.temperature = temperature
        
        # Thiết lập device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DQN-MCTS Agent using device: {self.device}")
        
        # Nếu không cung cấp DQN agent, tạo một agent mới
        if dqn_agent is None:
            self.dqn_agent = RLAgent(player_id, board_size=board_size)
            if model_path and os.path.exists(model_path):
                self.dqn_agent.load(model_path)
                print(f"Đã tải mô hình DQN từ {model_path}")
        else:
            self.dqn_agent = dqn_agent
        
        # Khởi tạo MCTS sử dụng mô hình DQN
        self.mcts = DQNMCTS(
            dqn_model=self.dqn_agent.policy_net,
            player_id=player_id,
            board_size=board_size,
            num_simulations=num_simulations,
            temperature=temperature,
            dirichlet_noise=False,  # Không sử dụng nhiễu khi chơi
            device=self.device
        )
    
    def get_action(self, game_state: Dict[str, Any]) -> Tuple[int, int]:
        """
        Lấy nước đi tốt nhất cho trạng thái hiện tại
        
        Args:
            game_state: Trạng thái trò chơi hiện tại
            
        Returns:
            Tuple[int, int]: Tọa độ (x, y) của nước đi được chọn
        """
        valid_moves = game_state['valid_moves']
        
        if not valid_moves:
            raise ValueError("Không có nước đi hợp lệ")
        
        # Nếu chỉ có một nước đi hợp lệ, chọn nước đó luôn
        if len(valid_moves) == 1:
            return valid_moves[0]
        
        # Thực hiện tìm kiếm MCTS
        action_probs = self.mcts.search(game_state)
        
        # Chọn nước đi theo phân phối xác suất
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        
        # Hiển thị thông tin phân tích
        if len(actions) > 0:
            # Sắp xếp các nước đi theo xác suất giảm dần
            sorted_actions = [(a, p) for a, p in zip(actions, probs)]
            sorted_actions.sort(key=lambda x: x[1], reverse=True)
            
            # Hiển thị top 3 nước đi tốt nhất
            print("\n[PHÂN TÍCH QUYẾT ĐỊNH CỦA DQN-MCTS]")
            for i, (action, prob) in enumerate(sorted_actions[:3]):
                print(f"  Lựa chọn #{i+1}: {action} với xác suất = {prob:.6f}")
        
        # Chọn nước đi có xác suất cao nhất
        best_action = max(action_probs.items(), key=lambda x: x[1])[0]
        return best_action
