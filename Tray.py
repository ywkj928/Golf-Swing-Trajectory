import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import cv2
import json
import matplotlib.pyplot as plt
import os
import glob
from ultralytics import YOLO

# 1. 세그멘테이션 모델 로드 및 좌표 추출
def load_custom_model(model_path):
    # YOLO 모델 로드
    model = YOLO(model_path)
    return model

def extract_coordinates_from_mask(mask):
    num_labels, labels = cv2.connectedComponents(mask)
    if num_labels <= 1:
        return None
    coords = np.where(labels == 1)
    if len(coords[0]) == 0:
        return None
    y_center = int(np.mean(coords[0]))
    x_center = int(np.mean(coords[1]))
    return [x_center, y_center]

def process_segmentation_output(model, frames):
    trajectory_coords = []
    for frame in frames:
        results = model(frame)
        for result in results:
            if result.masks is not None:
                mask = result.masks.data[0].cpu().numpy()
                mask = (mask > 0.5).astype(np.uint8)
                coords = extract_coordinates_from_mask(mask)
                if coords:
                    trajectory_coords.append(coords)
                    break
    return trajectory_coords

# 2. JSON으로 저장
def save_to_json(coords, filename="trajectory_data.json"):
    data = {
        "trajectory": [
            {"frame": i, "x": coord[0], "y": coord[1]}
            for i, coord in enumerate(coords)
        ]
    }
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Saved to {filename}")

# 3. JSON에서 데이터 로드
def load_from_json(filename="trajectory_data.json"):
    with open(filename, 'r') as f:
        data = json.load(f)
    coords = [[item["x"], item["y"]] for item in data["trajectory"]]
    return np.array(coords)

# 4. LSTM 데이터 준비
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# 5. LSTM 모델 정의
class GolfTrajectoryLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=50, num_layers=2):
        super(GolfTrajectoryLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# 6. 비디오에서 프레임 추출
def extract_frames_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'")
        return []
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

# 7. 메인 실행 함수 (스윙 궤도 예측 추가)
def main(model_path, frames, save_model_path="lstm_model.pth"):
    if not os.path.exists(model_path):
        print(f"Error: Model path '{model_path}' does not exist.")
        return
    segmentation_model = load_custom_model(model_path)
    coords = process_segmentation_output(segmentation_model, frames)
    if not coords:
        print("No coordinates extracted from segmentation.")
        return
    
    save_to_json(coords)
    raw_data = load_from_json()
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(raw_data)

    sequence_length = 3
    X, y = create_sequences(scaled_data, sequence_length)
    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = X.to(device)
    y = y.to(device)

    model = GolfTrajectoryLSTM(input_size=2, hidden_size=50, num_layers=2).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    num_epochs = 200
    for epoch in range(num_epochs):
        outputs = model(X)
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # 모델 저장
    torch.save(model.state_dict(), save_model_path)
    print(f"Model saved to {save_model_path}")

    model.eval()
    with torch.no_grad():
        last_sequence = X[-1].unsqueeze(0)
        predicted = model(last_sequence).cpu().numpy()
        predicted = scaler.inverse_transform(predicted)

    original_data = scaler.inverse_transform(scaled_data)
    plt.plot(original_data[:, 0], original_data[:, 1], label='Original Trajectory', marker='o')
    plt.plot([original_data[-1, 0], predicted[0, 0]], 
             [original_data[-1, 1], predicted[0, 1]], 
             label='Predicted', marker='x', color='red')
    plt.legend()
    plt.title('Golf Swing Trajectory Prediction')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()

# 8. 실행 예시
if __name__ == "__main__":
    model_path = input("Enter the path to your custom segmentation model: ")
    folder_path = input("Enter the folder path containing video files: ")
    video_files = glob.glob(os.path.join(folder_path, "*.mp4"))
    
    if not video_files:
        print("No video files found in the provided folder.")
    else:
        for video_file in video_files:
            print(f"Processing video: {video_file}")
            frames = extract_frames_from_video(video_file)
            if not frames:
                continue
            main(model_path, frames)
