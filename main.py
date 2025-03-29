import os
import sys
import json
import time
import argparse
import pygame
import numpy as np
import torch
from typing import Tuple, Dict, List, Any
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from game.board import Board
from game.game import GomokuGame
from game.rendering import Renderer
from agents.rl_agent import RLAgent
from utils.utils import set_seed, load_config

def play_against_ai(agent_path: str = None, player_first: bool = True, 
                   board_size: int = 15, win_length: int = 5) -> None:
    """
    Chơi game caro chống lại AI
    
    Args:
        agent_path: Đường dẫn đến mô hình đã huấn luyện
        player_first: True nếu người chơi đi trước (quân đen)
        board_size: Kích thước bàn cờ
        win_length: Số quân liên tiếp để chiến thắng
    """
    # Khởi tạo trò chơi
    game = GomokuGame(board_size, win_length)
    renderer = Renderer(board_size)
    
    # Khởi tạo agent
    player_id = Board.BLACK if player_first else Board.WHITE
    ai_id = Board.WHITE if player_first else Board.BLACK
    
    agent = RLAgent(ai_id, board_size=board_size)
    
    if agent_path and os.path.exists(agent_path):
        agent.load(agent_path)
        print(f"Đã tải mô hình AI. Epsilon hiện tại: {agent.epsilon:.4f}")
    else:
        print("Cảnh báo: Không tìm thấy mô hình. AI sẽ đi ngẫu nhiên.")
    
    # Render game
    game_state = game.get_state()
    renderer.render_board(game_state['board'], game.board.last_move)
    
    # Game loop
    running = True
    clock = pygame.time.Clock()
    
    # Nếu AI đi trước
    if not player_first and not game.is_game_over():
        print("AI đang suy nghĩ...")
        ai_action = agent.get_action(game.get_state())
        print(f"AI đi: ({ai_action[0]}, {ai_action[1]})")
        game.make_move(ai_action[0], ai_action[1])
        renderer.render_board(game.get_state()['board'], game.board.last_move)
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # Người chơi đi
            if event.type == pygame.MOUSEBUTTONDOWN and not game.is_game_over() and game.current_player == player_id:
                x, y = event.pos
                # Chuyển đổi tọa độ chuột sang tọa độ bàn cờ (điều chỉnh cho ô vuông)
                # Trừ margin và chia cho cell_size để xác định ô (không còn là điểm giao)
                col = int((x - renderer.margin) / renderer.cell_size)
                row = int((y - renderer.margin) / renderer.cell_size)
                
                if 0 <= row < board_size and 0 <= col < board_size:
                    if game.make_move(row, col):
                        renderer.render_board(game.get_state()['board'], game.board.last_move)
                        
                        # Kiểm tra kết thúc
                        if game.is_game_over():
                            renderer.render_game_over(game.get_winner())
                        else:
                            # AI đi
                            print("AI đang suy nghĩ...")
                            ai_action = agent.get_action(game.get_state())
                            # Hiển thị nước đi của AI
                            print(f"AI đi: ({ai_action[0]}, {ai_action[1]})")
                            
                            game.make_move(ai_action[0], ai_action[1])
                            renderer.render_board(game.get_state()['board'], game.board.last_move)
                            
                            # Kiểm tra kết thúc
                            if game.is_game_over():
                                renderer.render_game_over(game.get_winner())
        
        clock.tick(30)
    
    # Đóng game
    renderer.close()

def self_train(total_episodes: int = 10000, save_interval: int = 100, 
               model_path: str = None, resume: bool = False, 
               use_tensorboard: bool = True) -> None:
    """
    Tự động huấn luyện AI thông qua tự chơi
    
    Args:
        total_episodes: Tổng số tập huấn luyện
        save_interval: Số tập giữa mỗi lần lưu và đánh giá 
        model_path: Đường dẫn để lưu/tải mô hình
        resume: True để tiếp tục từ mô hình đã lưu
        use_tensorboard: Sử dụng TensorBoard để theo dõi quá trình huấn luyện
    """
    # Tải cấu hình
    config = load_config("config.json")
    board_size = config['game']['board_size']
    win_length = config['game']['win_length']
    
    # Đặt đường dẫn mô hình mặc định
    if model_path is None:
        model_path = f"saved_models/rl_self_train.pt"
        
    # Đảm bảo thư mục tồn tại
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Khởi tạo TensorBoard
    log_dir = f"logs/self_train_{time.strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(log_dir) if use_tensorboard else None
    
    # Khởi tạo agent
    agent = RLAgent(
        player_id=Board.BLACK,
        board_size=board_size,
        learning_rate=config['training']['learning_rate'],
        gamma=config['training']['gamma'],
        epsilon_start=config['training']['epsilon_start'] if not resume else 0.2,
        epsilon_end=config['training']['epsilon_end'],
        epsilon_decay=config['training']['epsilon_decay'],
        memory_size=config['training']['memory_size'],
        batch_size=config['training']['batch_size']
    )
    
    # Tải mô hình hiện có nếu resume=True
    if resume and os.path.exists(model_path):
        print(f"Đang tải mô hình từ {model_path}...")
        agent.load(model_path)
    
    # Thống kê huấn luyện
    stats = {
        'wins_black': 0,
        'wins_white': 0,
        'draws': 0,
        'avg_length': 0,
        'total_moves': 0,
        'episodes_done': 0,
        'start_time': time.time(),
        'win_rates': [],
        'episode_lengths': []
    }
    
    print("\n==== BẮT ĐẦU TỰ HUẤN LUYỆN ====")
    print(f"Tổng số tập: {total_episodes}")
    print(f"Lưu mô hình mỗi {save_interval} tập")
    print(f"Sử dụng TensorBoard: {use_tensorboard}")
    print(f"Đường dẫn mô hình: {model_path}")
    print("CHÚ Ý: CẦN HUẤN LUYỆN ÍT NHẤT 5000-10000 TẬP ĐỂ AI CHƠI TỐT!")
    print(f"Hiện tại bạn chỉ huấn luyện {total_episodes} tập, có thể chưa đủ.")
    
    if torch.cuda.is_available():
        print(f"CUDA sẵn sàng: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA không sẵn sàng, sử dụng CPU")
    
    # Main training loop
    try:
        for episode in tqdm(range(total_episodes)):
            # Khởi tạo trò chơi mới
            game = GomokuGame(board_size, win_length)
            
            # Lưu trữ trải nghiệm cho cả hai người chơi
            experiences = {
                Board.BLACK: [],
                Board.WHITE: []
            }
            
            # Chơi cho đến khi kết thúc
            move_count = 0
            
            while not game.is_game_over():
                current_state = game.get_state()
                current_player = current_state['current_player']
                
                # Lấy hành động từ agent (luôn sử dụng cùng một agent)
                action = agent.get_action(current_state)
                
                # Lưu trạng thái hiện tại và hành động
                experiences[current_player].append((current_state, action))
                
                # Thực hiện hành động
                game.make_move(action[0], action[1])
                move_count += 1
            
            # Cập nhật thống kê
            stats['total_moves'] += move_count
            stats['episodes_done'] += 1
            stats['avg_length'] = stats['total_moves'] / stats['episodes_done']
            stats['episode_lengths'].append(move_count)
            
            # Xử lý kết quả và phần thưởng
            winner = game.get_winner()
            
            if winner == Board.BLACK:
                stats['wins_black'] += 1
                reward_black = 1.0
                reward_white = -1.0
            elif winner == Board.WHITE:
                stats['wins_white'] += 1
                reward_black = -1.0
                reward_white = 1.0
            else:  # Hòa
                stats['draws'] += 1
                reward_black = 0.1
                reward_white = 0.1
            
            # Cập nhật model từ trải nghiệm
            final_state = game.get_state()
            
            # Cập nhật quân đen
            for i, (state, action) in enumerate(experiences[Board.BLACK]):
                # Nếu đây là trạng thái cuối cùng
                if i == len(experiences[Board.BLACK]) - 1:
                    next_state = final_state
                    done = True
                else:
                    # Lấy trạng thái sau khi đối thủ đi
                    next_idx = min(i + 1, len(experiences[Board.WHITE]) - 1)
                    next_state = experiences[Board.WHITE][next_idx][0] if experiences[Board.WHITE] else final_state
                    done = False
                
                # Cập nhật agent
                agent.update(state, action, reward_black if done else 0, next_state, done)
            
            # Cập nhật quân trắng
            for i, (state, action) in enumerate(experiences[Board.WHITE]):
                # Nếu đây là trạng thái cuối cùng
                if i == len(experiences[Board.WHITE]) - 1:
                    next_state = final_state
                    done = True
                else:
                    # Lấy trạng thái sau khi đối thủ đi
                    next_state = experiences[Board.BLACK][i+1][0]
                    done = False
                
                # Cập nhật agent
                agent.update(state, action, reward_white if done else 0, next_state, done)
            
            # Tính tỷ lệ thắng của quân đen
            black_win_rate = stats['wins_black'] / stats['episodes_done']
            stats['win_rates'].append(black_win_rate)
            
            # Log to TensorBoard
            if use_tensorboard and writer:
                writer.add_scalar('Wins/Black', stats['wins_black'], episode)
                writer.add_scalar('Wins/White', stats['wins_white'], episode)
                writer.add_scalar('Draws', stats['draws'], episode)
                writer.add_scalar('WinRate/Black', black_win_rate, episode)
                writer.add_scalar('Game/Length', move_count, episode)
                writer.add_scalar('Training/Epsilon', agent.epsilon, episode)
                
                if episode % 10 == 0 and agent.training_info['losses']:
                    writer.add_scalar('Training/Loss', agent.training_info['losses'][-1], episode)
                if episode % 10 == 0 and agent.training_info['q_values']:
                    writer.add_scalar('Training/QValue', sum(agent.training_info['q_values'][-10:]) / 10, episode)
            
            # Log to console
            if (episode + 1) % 100 == 0:
                elapsed_time = time.time() - stats['start_time']
                hours, remainder = divmod(elapsed_time, 3600)
                minutes, seconds = divmod(remainder, 60)
                
                tqdm.write(f"\nTập {episode+1}/{total_episodes} | Thời gian: {int(hours)}h {int(minutes)}m {int(seconds)}s")
                tqdm.write(f"Quân đen thắng: {stats['wins_black']} ({black_win_rate:.4f})")
                tqdm.write(f"Quân trắng thắng: {stats['wins_white']} ({stats['wins_white']/stats['episodes_done']:.4f})")
                tqdm.write(f"Hòa: {stats['draws']} ({stats['draws']/stats['episodes_done']:.4f})")
                tqdm.write(f"Độ dài trung bình: {stats['avg_length']:.1f} lượt")
                tqdm.write(f"Epsilon: {agent.epsilon:.6f}")
                if agent.training_info['losses']:
                    tqdm.write(f"Loss gần đây: {agent.training_info['losses'][-1]:.6f}")
            
            # Lưu mô hình định kỳ
            if (episode + 1) % save_interval == 0:
                checkpoint_path = f"{os.path.splitext(model_path)[0]}_ep{episode+1}.pt"
                agent.save(checkpoint_path)
                agent.save(model_path)  # Lưu mô hình chính
                
                # Vẽ biểu đồ tiến trình
                if (episode + 1) % (save_interval * 10) == 0:
                    plot_path = f"{os.path.splitext(model_path)[0]}_progress_ep{episode+1}.png"
                    plot_training_progress(stats, agent.training_info, plot_path)
    
    except KeyboardInterrupt:
        print("\nHuấn luyện bị ngắt bởi người dùng")
    
    # Lưu mô hình cuối cùng
    final_model_path = model_path
    agent.save(final_model_path)
    
    # Vẽ biểu đồ tiến trình cuối cùng
    plot_path = f"{os.path.splitext(model_path)[0]}_progress_final.png"
    plot_training_progress(stats, agent.training_info, plot_path)
    
    # Đóng TensorBoard writer
    if writer:
        writer.close()
    
    print("\n==== HOÀN THÀNH TỰ HUẤN LUYỆN ====")
    print(f"Tổng số tập đã huấn luyện: {stats['episodes_done']}")
    print(f"Tỷ lệ thắng cuối cùng (quân đen): {stats['wins_black']/stats['episodes_done']:.4f}")
    print(f"Độ dài trò chơi trung bình: {stats['avg_length']:.1f}")
    print(f"Mô hình cuối cùng đã được lưu tại: {final_model_path}")
    print(f"Biểu đồ tiến trình đã được lưu tại: {plot_path}")
    if use_tensorboard:
        print(f"Nhật ký TensorBoard đã được lưu tại: {log_dir}")
        print(f"Để xem dữ liệu trong TensorBoard, chạy: tensorboard --logdir={log_dir}")

def plot_training_progress(stats: Dict[str, Any], training_info: Dict[str, List[float]], 
                          save_path: str = None) -> None:
    """
    Vẽ biểu đồ tiến trình huấn luyện
    
    Args:
        stats: Thống kê về các trận đấu
        training_info: Thông tin huấn luyện từ agent
        save_path: Đường dẫn để lưu biểu đồ
    """
    plt.figure(figsize=(20, 15))
    
    # Plot 1: Win rate
    plt.subplot(3, 2, 1)
    plt.plot(stats['win_rates'], 'b-', label='Tỷ lệ thắng (Quân đen)')
    plt.title('Tỷ lệ thắng theo thời gian')
    plt.xlabel('Tập')
    plt.ylabel('Tỷ lệ thắng')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Game length
    plt.subplot(3, 2, 2)
    plt.plot(stats['episode_lengths'], 'g-', alpha=0.5)
    # Đường trung bình động
    window_size = min(50, len(stats['episode_lengths']))
    if window_size > 0:
        moving_avg = np.convolve(stats['episode_lengths'], np.ones(window_size)/window_size, mode='valid')
        plt.plot(moving_avg, 'r-', label=f'Trung bình động (window={window_size})')
    plt.title('Độ dài trò chơi')
    plt.xlabel('Tập')
    plt.ylabel('Số lượt')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Epsilon decay
    plt.subplot(3, 2, 3)
    if training_info['epsilons']:
        plt.plot(training_info['epsilons'], 'm-')
        plt.title('Epsilon theo thời gian')
        plt.xlabel('Bước cập nhật')
        plt.ylabel('Epsilon')
        plt.grid(True)
    
    # Plot 4: Loss
    plt.subplot(3, 2, 4)
    if training_info['losses']:
        plt.plot(training_info['losses'], 'r-', alpha=0.5)
        # Đường trung bình động cho loss
        window_size = min(100, len(training_info['losses']))
        if window_size > 0:
            moving_avg = np.convolve(training_info['losses'], np.ones(window_size)/window_size, mode='valid')
            plt.plot(moving_avg, 'b-', label=f'Trung bình động (window={window_size})')
        plt.title('Loss theo thời gian')
        plt.xlabel('Bước cập nhật')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
    
    # Plot 5: Q-values
    plt.subplot(3, 2, 5)
    if training_info['q_values']:
        plt.plot(training_info['q_values'], 'c-', alpha=0.5)
        # Đường trung bình động cho Q-values
        window_size = min(100, len(training_info['q_values']))
        if window_size > 0:
            moving_avg = np.convolve(training_info['q_values'], np.ones(window_size)/window_size, mode='valid')
            plt.plot(moving_avg, 'k-', label=f'Trung bình động (window={window_size})')
        plt.title('Q-values theo thời gian')
        plt.xlabel('Bước cập nhật')
        plt.ylabel('Q-value')
        plt.legend()
        plt.grid(True)
    
    # Plot 6: Rewards
    plt.subplot(3, 2, 6)
    if training_info['rewards']:
        plt.plot(training_info['rewards'], 'y-', alpha=0.5)
        # Đường trung bình động cho rewards
        window_size = min(100, len(training_info['rewards']))
        if window_size > 0:
            moving_avg = np.convolve(training_info['rewards'], np.ones(window_size)/window_size, mode='valid')
            plt.plot(moving_avg, 'k-', label=f'Trung bình động (window={window_size})')
        plt.title('Phần thưởng theo thời gian')
        plt.xlabel('Bước cập nhật')
        plt.ylabel('Phần thưởng')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def check_cuda_compatibility():
    """
    Kiểm tra khả năng tương thích với CUDA
    """
    print("\n==== KIỂM TRA TƯƠNG THÍCH CUDA ====")
    
    print(f"PyTorch version: {torch.__version__}")
    
    if torch.cuda.is_available():
        print("CUDA is available!")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        
        # Test tensor on GPU
        try:
            x = torch.rand(5, 5).cuda()
            y = torch.rand(5, 5).cuda()
            z = x + y
            print("GPU computation test: Passed!")
        except Exception as e:
            print(f"GPU computation test: Failed! Error: {e}")
    else:
        print("CUDA is not available. Using CPU instead.")
        print("To enable CUDA, make sure you have installed:")
        print("1. NVIDIA GPU drivers")
        print("2. CUDA Toolkit 12.x")
        print("3. PyTorch with CUDA support")

def main():
    parser = argparse.ArgumentParser(description='Gomoku (Caro) AI Self-Learning')
    
    subparsers = parser.add_subparsers(dest='command')
    
    # Play command
    play_parser = subparsers.add_parser('play', help='Chơi caro chống lại AI')
    play_parser.add_argument('--model', type=str, default='saved_models/rl_self_train.pt', 
                            help='Đường dẫn đến mô hình đã huấn luyện')
    play_parser.add_argument('--ai-first', action='store_true', help='AI đi trước')
    
    # Self-train command
    train_parser = subparsers.add_parser('train', help='Tự huấn luyện AI')
    train_parser.add_argument('--episodes', type=int, default=10000, help='Số tập huấn luyện')
    train_parser.add_argument('--save-interval', type=int, default=100, help='Số tập giữa mỗi lần lưu')
    train_parser.add_argument('--model', type=str, default='saved_models/rl_self_train.pt', 
                             help='Đường dẫn để lưu/tải mô hình')
    train_parser.add_argument('--resume', action='store_true', help='Tiếp tục huấn luyện từ mô hình đã lưu')
    train_parser.add_argument('--no-tensorboard', action='store_true', help='Tắt TensorBoard logging')
    
    # CUDA check command
    cuda_parser = subparsers.add_parser('cuda-check', help='Kiểm tra tương thích CUDA')
    
    # Seed
    parser.add_argument('--seed', type=int, help='Seed cho tính ngẫu nhiên')
    
    args = parser.parse_args()
    
    # Đặt seed
    if args.seed is not None:
        set_seed(args.seed)
    
    if args.command == 'play':
        play_against_ai(args.model, not args.ai_first)
    elif args.command == 'train':
        self_train(args.episodes, args.save_interval, args.model, args.resume, not args.no_tensorboard)
    elif args.command == 'cuda-check':
        check_cuda_compatibility()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
