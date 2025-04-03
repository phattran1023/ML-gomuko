import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import random
from typing import Dict, List, Tuple, Any
from tqdm import tqdm

from game.board import Board
from game.game import GomokuGame
from agents.rl_agent import RLAgent
from agents.dqn_mcts_agent import DQNMCTSAgent
from utils.utils import load_config, set_seed

class RandomAgent:
    """Agent đưa ra quyết định ngẫu nhiên, dùng để đánh giá mô hình"""
    def __init__(self, player_id):
        self.player_id = player_id
        
    def get_action(self, game_state):
        valid_moves = game_state['valid_moves']
        # Sửa lỗi: Dùng random.choice thay vì np.random.choice để tránh lỗi "a must be 1-dimensional"
        return random.choice(valid_moves) if valid_moves else None

def evaluate_against_random(model_path: str, num_games: int = 100, 
                           board_size: int = 15, model_plays_first: bool = True, 
                           use_mcts: bool = False, num_simulations: int = 100) -> Dict:
    """
    Đánh giá mô hình bằng cách đối đầu với agent ngẫu nhiên
    
    Args:
        model_path: Đường dẫn đến mô hình cần đánh giá
        num_games: Số trận đấu cần đánh giá
        board_size: Kích thước bàn cờ
        model_plays_first: True nếu mô hình đi trước
        use_mcts: Sử dụng MCTS kết hợp với DQN
        num_simulations: Số lần mô phỏng cho mỗi nước đi MCTS
        
    Returns:
        Dict: Thống kê kết quả đánh giá
    """
    # Thiết lập agent
    model_id = Board.BLACK if model_plays_first else Board.WHITE
    random_id = Board.WHITE if model_plays_first else Board.BLACK
    
    # Tạo agent
    if use_mcts:
        model_agent = DQNMCTSAgent(model_id, model_path=model_path, 
                                  board_size=board_size, num_simulations=num_simulations)
        agent_name = "DQN-MCTS"
    else:
        model_agent = RLAgent(model_id, board_size=board_size)
        model_agent.load(model_path)
        # Đặt epsilon = 0 để luôn chọn nước đi tốt nhất theo mô hình
        model_agent.epsilon = 0.0
        agent_name = "DQN"
    
    random_agent = RandomAgent(random_id)
    
    # Thống kê
    stats = {
        'model_wins': 0,
        'random_wins': 0,
        'draws': 0,
        'avg_game_length': 0,
        'model_avg_q_value': 0.0,
        'total_moves': 0,
        'q_values': []
    }
    
    # Đánh giá mô hình qua nhiều trận đấu
    all_q_values = []
    
    print(f"Đánh giá {agent_name} qua {num_games} trận đấu với agent ngẫu nhiên...")
    print(f"Mô hình: {model_path}")
    print(f"Mô hình đi {'trước' if model_plays_first else 'sau'}")
    if use_mcts:
        print(f"Số lần mô phỏng MCTS: {num_simulations}")
    
    for game_idx in tqdm(range(num_games)):
        game = GomokuGame(board_size)
        moves = 0
        
        # Game loop
        while not game.is_game_over():
            state = game.get_state()
            current_player = state['current_player']
            
            if current_player == model_id:
                # Mô hình đưa ra quyết định
                if isinstance(model_agent, RLAgent):
                    # Lưu lại Q-value cho phân tích
                    with torch.no_grad():
                        state_tensor = model_agent._preprocess_state(state)
                        q_values, _ = model_agent.policy_net(state_tensor)
                        
                        # Lọc chỉ các nước đi hợp lệ
                        valid_moves = state['valid_moves']
                        valid_mask = torch.zeros(board_size * board_size, device=model_agent.device)
                        for x, y in valid_moves:
                            idx = x * board_size + y
                            valid_mask[idx] = 1
                        
                        masked_q_values = q_values.squeeze(0) * valid_mask
                        max_q_value = masked_q_values.max().item()
                        all_q_values.append(max_q_value)
                
                action = model_agent.get_action(state)
            else:
                # Agent ngẫu nhiên đưa ra quyết định
                action = random_agent.get_action(state)
            
            # Thực hiện nước đi
            game.make_move(action[0], action[1])
            moves += 1
        
        # Cập nhật thống kê
        winner = game.get_winner()
        if winner == model_id:
            stats['model_wins'] += 1
        elif winner == random_id:
            stats['random_wins'] += 1
        else:
            stats['draws'] += 1
        
        stats['total_moves'] += moves
    
    # Tính toán thống kê tổng thể
    stats['avg_game_length'] = stats['total_moves'] / num_games
    stats['win_rate'] = stats['model_wins'] / num_games
    
    if all_q_values:
        stats['model_avg_q_value'] = sum(all_q_values) / len(all_q_values)
        stats['q_values'] = all_q_values
    
    return stats

def evaluate_winning_positions(model_path: str, num_positions: int = 100, 
                              board_size: int = 15, use_mcts: bool = False,
                              num_simulations: int = 100) -> Dict:
    """
    Đánh giá khả năng phát hiện nước đi chiến thắng của mô hình
    
    Args:
        model_path: Đường dẫn đến mô hình cần đánh giá
        num_positions: Số vị trí cần đánh giá
        board_size: Kích thước bàn cờ
        use_mcts: Sử dụng MCTS kết hợp với DQN
        num_simulations: Số lần mô phỏng cho mỗi nước đi MCTS
        
    Returns:
        Dict: Thống kê kết quả đánh giá
    """
    # Tạo agent đánh giá
    player_id = Board.BLACK  # Luôn đánh giá từ góc nhìn quân đen
    
    if use_mcts:
        agent = DQNMCTSAgent(player_id, model_path=model_path, 
                             board_size=board_size, num_simulations=num_simulations)
        agent_name = "DQN-MCTS"
    else:
        agent = RLAgent(player_id, board_size=board_size)
        agent.load(model_path)
        agent.epsilon = 0.0  # Luôn chọn nước đi tốt nhất
        agent_name = "DQN"
    
    # Thống kê
    stats = {
        'winning_positions_found': 0,
        'avg_q_value_winning_moves': 0.0,
        'q_values_winning': [],
        'q_values_other': []
    }
    
    print(f"Đánh giá khả năng phát hiện nước đi chiến thắng của {agent_name}...")
    print(f"Mô hình: {model_path}")
    print(f"Số vị trí kiểm tra: {num_positions}")
    
    # Khởi tạo trò chơi
    all_winning_q_values = []
    all_other_q_values = []
    
    # Tạo vị trí ngẫu nhiên có nước đi chiến thắng
    positions_tested = 0
    positions_with_winning_move = 0
    
    with tqdm(total=num_positions) as pbar:
        while positions_tested < num_positions:
            # Khởi tạo trò chơi mới
            game = GomokuGame(board_size)
            
            # Chơi ngẫu nhiên một số nước để tạo vị trí nửa chừng (10-40 nước)
            moves_to_play = np.random.randint(10, 40)
            
            for _ in range(moves_to_play):
                if game.is_game_over():
                    break
                    
                valid_moves = game.get_state()['valid_moves']
                if not valid_moves:
                    break
                    
                action = valid_moves[np.random.randint(0, len(valid_moves))]
                game.make_move(action[0], action[1])
            
            if game.is_game_over():
                continue  # Trò chơi đã kết thúc, thử lại
            
            # Kiểm tra xem có nước đi nào dẫn đến chiến thắng ngay không
            state = game.get_state()
            current_player = state['current_player']
            valid_moves = state['valid_moves']
            
            winning_moves = []
            other_moves = []
            
            # Kiểm tra từng nước đi hợp lệ
            for move in valid_moves:
                # Tạo bản sao trò chơi để thử nước đi
                test_game = game.clone()
                test_game.make_move(move[0], move[1])
                
                if test_game.is_game_over() and test_game.get_winner() == current_player:
                    winning_moves.append(move)
                else:
                    other_moves.append(move)
            
            # Có ít nhất một nước chiến thắng
            if winning_moves:
                positions_with_winning_move += 1
                
                # Lấy hành động từ mô hình
                action = agent.get_action(state)
                
                # Kiểm tra xem action có phải nước chiến thắng
                is_winning_move = action in winning_moves
                
                if is_winning_move:
                    stats['winning_positions_found'] += 1
                
                # Lấy Q-values cho phân tích
                if isinstance(agent, RLAgent):
                    with torch.no_grad():
                        state_tensor = agent._preprocess_state(state)
                        q_values, _ = agent.policy_net(state_tensor)
                        q_values = q_values.squeeze(0).cpu().numpy()
                        
                        # Q-values cho nước thắng
                        for move in winning_moves:
                            idx = move[0] * board_size + move[1]
                            all_winning_q_values.append(q_values[idx])
                        
                        # Q-values cho nước khác
                        for move in other_moves:
                            idx = move[0] * board_size + move[1]
                            all_other_q_values.append(q_values[idx])
                
                positions_tested += 1
                pbar.update(1)
    
    # Tính toán thống kê
    if positions_with_winning_move > 0:
        stats['winning_move_accuracy'] = stats['winning_positions_found'] / positions_with_winning_move
    else:
        stats['winning_move_accuracy'] = 0.0
    
    if all_winning_q_values:
        stats['avg_q_value_winning_moves'] = sum(all_winning_q_values) / len(all_winning_q_values)
        stats['q_values_winning'] = all_winning_q_values
    
    if all_other_q_values:
        stats['avg_q_value_other_moves'] = sum(all_other_q_values) / len(all_other_q_values)
        stats['q_values_other'] = all_other_q_values
    
    return stats

def compare_models(model_paths: List[str], num_games: int = 50, 
                  board_size: int = 15, use_mcts: bool = False,
                  num_simulations: int = 100) -> Dict:
    """
    So sánh nhiều mô hình với nhau
    
    Args:
        model_paths: Danh sách đường dẫn đến các mô hình cần so sánh
        num_games: Số trận đấu cho mỗi cặp mô hình
        board_size: Kích thước bàn cờ
        use_mcts: Sử dụng MCTS kết hợp với DQN
        num_simulations: Số lần mô phỏng cho mỗi nước đi MCTS
        
    Returns:
        Dict: Thống kê kết quả so sánh
    """
    num_models = len(model_paths)
    results = np.zeros((num_models, num_models))
    
    agent_type = "DQN-MCTS" if use_mcts else "DQN"
    print(f"So sánh {num_models} mô hình sử dụng {agent_type}...")
    
    all_agents = []
    
    # Khởi tạo các agent
    for i, path in enumerate(model_paths):
        if use_mcts:
            agent = DQNMCTSAgent(Board.BLACK, model_path=path, 
                                board_size=board_size, num_simulations=num_simulations)
        else:
            agent = RLAgent(Board.BLACK, board_size=board_size)
            agent.load(path)
            agent.epsilon = 0.0
        all_agents.append(agent)
        print(f"- Mô hình {i+1}: {path}")
    
    for i in range(num_models):
        for j in range(i+1, num_models):
            print(f"\nĐánh giá: Mô hình {i+1} vs Mô hình {j+1} ({num_games} trận)")
            
            # Thống kê
            model_i_wins = 0
            model_j_wins = 0
            draws = 0
            
            for game_idx in tqdm(range(num_games)):
                # Mỗi mô hình sẽ được chơi cả hai phía (đen và trắng)
                game = GomokuGame(board_size)
                
                # Xác định ai đi trước (mô hình i hoặc j)
                black_idx = i if game_idx % 2 == 0 else j
                white_idx = j if game_idx % 2 == 0 else i
                
                # Game loop
                while not game.is_game_over():
                    state = game.get_state()
                    current_player = state['current_player']
                    
                    # Chọn agent tương ứng
                    if current_player == Board.BLACK:
                        agent = all_agents[black_idx]
                        agent.player_id = Board.BLACK
                    else:
                        agent = all_agents[white_idx]
                        agent.player_id = Board.WHITE
                    
                    # Lấy và thực hiện nước đi
                    action = agent.get_action(state)
                    game.make_move(action[0], action[1])
                
                # Xác định người thắng
                winner = game.get_winner()
                
                if winner == Board.BLACK:
                    if black_idx == i:
                        model_i_wins += 1
                    else:
                        model_j_wins += 1
                elif winner == Board.WHITE:
                    if white_idx == i:
                        model_i_wins += 1
                    else:
                        model_j_wins += 1
                else:
                    draws += 1
            
            # Cập nhật matrix kết quả
            results[i, j] = model_i_wins / num_games
            results[j, i] = model_j_wins / num_games
            
            print(f"Kết quả: Mô hình {i+1} thắng {model_i_wins}, Mô hình {j+1} thắng {model_j_wins}, Hòa {draws}")
            print(f"Tỷ lệ thắng: Mô hình {i+1}: {model_i_wins/num_games:.4f}, Mô hình {j+1}: {model_j_wins/num_games:.4f}")
    
    # Tính toán điểm tổng thể của mỗi mô hình
    model_scores = np.sum(results, axis=1)
    
    return {
        'results_matrix': results,
        'model_scores': model_scores,
        'model_paths': model_paths
    }

def plot_evaluation_results(random_stats, winning_stats, model_comparison=None, 
                           output_path="model_evaluation.png"):
    """
    Vẽ biểu đồ từ kết quả đánh giá
    
    Args:
        random_stats: Kết quả đánh giá với agent ngẫu nhiên
        winning_stats: Kết quả đánh giá nước đi chiến thắng
        model_comparison: Kết quả so sánh các mô hình (optional)
        output_path: Đường dẫn lưu biểu đồ
    """
    plt.figure(figsize=(20, 15))
    
    # Plot 1: Tỷ lệ thắng và độ dài trò chơi
    plt.subplot(2, 2, 1)
    labels = ['Thắng', 'Thua', 'Hòa']
    sizes = [random_stats['model_wins'], random_stats['random_wins'], random_stats['draws']]
    colors = ['#4CAF50', '#F44336', '#2196F3']
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Kết quả với Agent Ngẫu nhiên')
    
    # Plot 2: Q-values phân phối
    plt.subplot(2, 2, 2)
    if 'q_values' in random_stats and random_stats['q_values']:
        plt.hist(random_stats['q_values'], bins=20, color='#2196F3', alpha=0.7)
        plt.axvline(x=random_stats['model_avg_q_value'], color='r', linestyle='--', 
                  label=f'Trung bình: {random_stats["model_avg_q_value"]:.2f}')
        plt.title('Phân phối Q-values')
        plt.xlabel('Q-value')
        plt.ylabel('Tần suất')
        plt.legend()
    
    # Plot 3: Tỷ lệ phát hiện nước thắng
    plt.subplot(2, 2, 3)
    if 'winning_move_accuracy' in winning_stats:
        accuracy = winning_stats['winning_move_accuracy'] * 100
        plt.bar(['Phát hiện nước thắng'], [accuracy], color='#4CAF50')
        plt.bar(['Phát hiện nước thắng'], [100 - accuracy], bottom=[accuracy], color='#F44336')
        plt.ylim(0, 100)
        plt.title('Khả năng phát hiện nước chiến thắng')
        plt.ylabel('Tỉ lệ phần trăm (%)')
        
    # Plot 4: Q-values cho nước thắng và nước khác
    plt.subplot(2, 2, 4)
    if ('q_values_winning' in winning_stats and winning_stats['q_values_winning'] and
        'q_values_other' in winning_stats and winning_stats['q_values_other']):
        
        win_q = winning_stats['q_values_winning']
        other_q = winning_stats['q_values_other']
        
        plt.violinplot([win_q, other_q], showmeans=True)
        plt.xticks([1, 2], ['Nước thắng', 'Nước khác'])
        plt.title('So sánh Q-values: Nước thắng vs Nước khác')
        plt.ylabel('Q-value')
        
        # Thêm text hiển thị giá trị trung bình
        if 'avg_q_value_winning_moves' in winning_stats and 'avg_q_value_other_moves' in winning_stats:
            win_avg = winning_stats['avg_q_value_winning_moves']
            other_avg = winning_stats['avg_q_value_other_moves']
            plt.text(0.8, win_avg, f"{win_avg:.2f}", ha='center', va='bottom')
            plt.text(1.8, other_avg, f"{other_avg:.2f}", ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Đã lưu biểu đồ đánh giá tại: {output_path}")
    
def main():
    parser = argparse.ArgumentParser(description='Đánh giá mô hình AI Gomoku')
    
    parser.add_argument('--model', type=str, required=True,
                       help='Đường dẫn đến mô hình cần đánh giá')
    parser.add_argument('--mode', type=str, choices=['random', 'winning', 'compare', 'all'],
                       default='all', help='Chế độ đánh giá')
    parser.add_argument('--num-games', type=int, default=100,
                       help='Số trận đấu để đánh giá')
    parser.add_argument('--use-mcts', action='store_true',
                       help='Sử dụng MCTS kết hợp với DQN')
    parser.add_argument('--simulations', type=int, default=100,
                       help='Số lần mô phỏng MCTS')
    parser.add_argument('--models-dir', type=str, default='saved_models',
                       help='Thư mục chứa các mô hình cần so sánh')
    parser.add_argument('--output', type=str, default='model_evaluation.png',
                       help='Đường dẫn lưu biểu đồ đánh giá')
    
    args = parser.parse_args()
    
    # Tải cấu hình
    config = load_config("config.json")
    board_size = config['game']['board_size']
    
    # Đánh giá dựa trên mode
    if args.mode in ('random', 'all'):
        random_stats = evaluate_against_random(
            args.model, 
            num_games=args.num_games, 
            board_size=board_size,
            use_mcts=args.use_mcts,
            num_simulations=args.simulations
        )
        
        print("\n=== KẾT QUẢ ĐÁNH GIÁ VỚI AGENT NGẪU NHIÊN ===")
        print(f"Tổng số trận: {args.num_games}")
        print(f"Thắng: {random_stats['model_wins']} ({random_stats['win_rate']:.4f})")
        print(f"Thua: {random_stats['random_wins']} ({random_stats['random_wins']/args.num_games:.4f})")
        print(f"Hòa: {random_stats['draws']} ({random_stats['draws']/args.num_games:.4f})")
        print(f"Độ dài trung bình mỗi trận: {random_stats['avg_game_length']:.2f} nước")
        if 'model_avg_q_value' in random_stats:
            print(f"Q-value trung bình: {random_stats['model_avg_q_value']:.4f}")
    
    if args.mode in ('winning', 'all'):
        winning_stats = evaluate_winning_positions(
            args.model,
            num_positions=100,
            board_size=board_size,
            use_mcts=args.use_mcts,
            num_simulations=args.simulations
        )
        
        print("\n=== KẾT QUẢ ĐÁNH GIÁ KHẢ NĂNG PHÁT HIỆN NƯỚC THẮNG ===")
        if 'winning_move_accuracy' in winning_stats:
            print(f"Tỷ lệ phát hiện nước thắng: {winning_stats['winning_move_accuracy']:.4f}")
        if 'avg_q_value_winning_moves' in winning_stats:
            print(f"Q-value trung bình cho nước thắng: {winning_stats['avg_q_value_winning_moves']:.4f}")
        if 'avg_q_value_other_moves' in winning_stats:
            print(f"Q-value trung bình cho nước khác: {winning_stats['avg_q_value_other_moves']:.4f}")
    
    if args.mode in ('compare', 'all'):
        # Tìm tất cả các mô hình checkpoint trong thư mục
        model_paths = []
        if os.path.isdir(args.models_dir):
            for file in os.listdir(args.models_dir):
                if file.endswith('.pt') and '_ep' in file:
                    model_paths.append(os.path.join(args.models_dir, file))
        
        # Thêm mô hình hiện tại
        if args.model not in model_paths:
            model_paths.append(args.model)
        
        # Sắp xếp theo số episode
        def extract_ep_number(path):
            filename = os.path.basename(path)
            if '_ep' in filename:
                try:
                    return int(filename.split('_ep')[1].split('.')[0])
                except:
                    return 0
            return float('inf')  # Đặt mô hình không có số ep lên cuối
        
        model_paths.sort(key=extract_ep_number)
        
        # Nếu có nhiều hơn 5 mô hình, chỉ lấy 5 mô hình gần nhất
        if len(model_paths) > 5:
            model_paths = model_paths[-5:]
        
        if len(model_paths) > 1:
            comparison_stats = compare_models(
                model_paths,
                num_games=max(10, min(50, args.num_games // 2)),  # Điều chỉnh số lượng trận để tiết kiệm thời gian
                board_size=board_size,
                use_mcts=args.use_mcts,
                num_simulations=args.simulations
            )
            
            print("\n=== KẾT QUẢ SO SÁNH CÁC MÔ HÌNH ===")
            for i, score in enumerate(comparison_stats['model_scores']):
                path = comparison_stats['model_paths'][i]
                print(f"Mô hình {i+1} ({os.path.basename(path)}): Điểm = {score:.4f}")
        else:
            print("\nKhông đủ mô hình để so sánh. Cần ít nhất 2 mô hình.")
            comparison_stats = None
    
    # Vẽ biểu đồ nếu có đủ dữ liệu
    if args.mode == 'all':
        if 'random_stats' in locals() and 'winning_stats' in locals():
            plot_evaluation_results(
                random_stats, 
                winning_stats,
                output_path=args.output
            )

if __name__ == "__main__":
    main()
