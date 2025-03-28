# BotGomuko - AI Tự Học cho Game Caro (Gomoku)

BotGomuko là một codebase chuyên nghiệp được xây dựng để huấn luyện và phát triển AI tự học chơi cờ Caro (Gomoku). Codebase tập trung vào kỹ thuật Deep Reinforcement Learning và được tối ưu hóa để tận dụng CUDA 12.8 cho việc huấn luyện hiệu quả.

## Tính năng chính

- Game engine hoàn chỉnh với luật Gomoku tiêu chuẩn
- AI sử dụng Deep Reinforcement Learning (DQN) với các cải tiến:
  - Prioritized Experience Replay
  - Double DQN
  - Dueling DQN Architecture
  - Residual Connections
- Hỗ trợ đầy đủ CUDA 12.8 cho huấn luyện trên GPU
- Framework tự huấn luyện thông qua tự chơi
- Giao diện đồ họa cho phép chơi chống lại AI đã huấn luyện
- Công cụ phân tích và theo dõi quá trình huấn luyện với TensorBoard
- Tự động lưu và tiếp tục huấn luyện từ checkpoint

## Cài đặt

### Yêu cầu

- Python 3.7 trở lên
- CUDA 12.8 và GPU NVIDIA tương thích (khuyến nghị)
- Các thư viện được liệt kê trong `requirements.txt`

### Cài đặt các thư viện

```bash
pip install -r requirements.txt
```

## Kiểm tra tương thích CUDA

Trước khi bắt đầu huấn luyện, bạn nên kiểm tra tương thích CUDA:

```bash
python main.py cuda-check
```

## Cách sử dụng

BotGomuko cung cấp hai chức năng chính: tự huấn luyện AI và chơi chống lại AI đã huấn luyện.

### Tự huấn luyện AI

```bash
# Huấn luyện với 10,000 tập
python main.py train --episodes 10000

# Tiếp tục huấn luyện từ mô hình đã lưu
python main.py train --episodes 5000 --resume --model saved_models/my_model.pt

# Huấn luyện với cấu hình tùy chỉnh
python main.py train --episodes 20000 --save-interval 500 --model my_models/advanced_ai.pt
```

### Chơi chống lại AI

```bash
# Chơi với AI đã huấn luyện (người chơi đi trước)
python main.py play --model saved_models/rl_self_train.pt

# Chơi với AI đi trước
python main.py play --model saved_models/rl_self_train.pt --ai-first
```

## Theo dõi quá trình huấn luyện

BotGomuko sử dụng TensorBoard để theo dõi quá trình huấn luyện. Khi bắt đầu huấn luyện, TensorBoard logs sẽ được lưu trong thư mục `logs/`. Bạn có thể xem các chỉ số này bằng cách chạy:

```bash
tensorboard --logdir=logs
```

Sau đó mở trình duyệt web và truy cập `http://localhost:6006`

## Cách hoạt động của AI

AI trong BotGomuko sử dụng Deep Q-Learning với nhiều cải tiến:

1. **Model Architecture**: Sử dụng mạng CNN sâu với residual connections để xử lý thông tin bàn cờ
2. **Prioritized Experience Replay**: Ưu tiên học từ những trải nghiệm quan trọng
3. **Double DQN**: Giảm thiểu overestimation của Q-values
4. **Self-Play**: AI tự học bằng cách chơi với chính mình

Mỗi khi hoàn thành một trận đấu, AI sẽ cập nhật mô hình dựa trên kết quả và trải nghiệm thu được, giúp nó dần dần cải thiện chiến thuật.

## Lưu trữ và tải mô hình

Mô hình được lưu tự động trong quá trình huấn luyện vào thư mục `saved_models/`. Bạn có thể tiếp tục huấn luyện từ checkpoint hoặc sử dụng mô hình đã huấn luyện để chơi.

## Tối ưu hóa hiệu suất

- Đối với GPU có ít VRAM, hãy giảm `batch_size` trong file `config.json`
- Để tăng tốc độ huấn luyện, hãy tăng `batch_size` nếu GPU của bạn có đủ VRAM
- Điều chỉnh `epsilon_decay` để kiểm soát tốc độ chuyển từ thăm dò sang khai thác

## Tác giả

© 2023 BotGomuko Team
