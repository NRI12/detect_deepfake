import cv2
import torch
import numpy as np
from torchvision.transforms import ToTensor, Normalize, Compose
import face_recognition
from torch import nn
from torchvision import models

# Định nghĩa mô hình DeepfakeModel
class DeepfakeModel(nn.Module):
    def __init__(self, num_classes=2, lstm_hidden_dim=2048, lstm_layers=1, bidirectional=False):
        super(DeepfakeModel, self).__init__()
        base_model = models.resnext50_32x4d(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        feature_dim = 2048
        self.lstm = nn.LSTM(input_size=feature_dim,
                            hidden_size=lstm_hidden_dim,
                            num_layers=lstm_layers,
                            batch_first=True,
                            bidirectional=bidirectional)
        lstm_output_dim = lstm_hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(lstm_output_dim, num_classes)

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        x = x.view(batch_size * seq_len, c, h, w)
        x = self.feature_extractor(x)
        x = self.avgpool(x)
        x = x.view(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        out = self.fc(lstm_out)
        return out

# Tải mô hình đã lưu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepfakeModel().to(device)
model.load_state_dict(torch.load('deepfake_model.pth', map_location=device))
model.eval()
print("Model loaded and ready for inference")

# Hàm trích xuất khuôn mặt từ video
def extract_faces_from_video(video_path, output_size=(112, 112), num_frames=16):
    """
    Trích xuất khuôn mặt từ video và trả về các frame chỉ chứa khuôn mặt.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    success, frame = cap.read()
    
    while success and len(frames) < num_frames:
        # Chuyển frame sang RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Tìm khuôn mặt trong frame
        face_locations = face_recognition.face_locations(rgb_frame)
        
        for (top, right, bottom, left) in face_locations:
            # Cắt khuôn mặt
            face = rgb_frame[top:bottom, left:right]
            
            # Resize khuôn mặt về kích thước chuẩn
            face = cv2.resize(face, output_size)
            
            frames.append(face)
            break  # Chỉ lấy khuôn mặt đầu tiên mỗi frame
        
        success, frame = cap.read()
    
    cap.release()

    # Pad nếu số lượng frame ít hơn yêu cầu
    while len(frames) < num_frames:
        frames.append(np.zeros((output_size[0], output_size[1], 3), dtype=np.uint8))
    
    # Áp dụng transform
    transform = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    frames = torch.stack([transform(frame) for frame in frames])  # (num_frames, C, H, W)
    return frames.unsqueeze(0)  # (1, num_frames, C, H, W)

# Hàm dự đoán
def predict_with_faces(video_path):
    """
    Sử dụng mô hình để dự đoán nhãn video, sau khi trích xuất khuôn mặt.
    """
    # Trích xuất khuôn mặt từ video
    frames = extract_faces_from_video(video_path).to(device)

    # Dự đoán
    with torch.no_grad():
        logits = model(frames)
        probabilities = torch.softmax(logits, dim=1)
        predicted_label = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_label].item()

    label_name = "Real" if predicted_label == 1 else "Fake"
    return label_name, confidence

# Đường dẫn video cần dự đoán
video_path = r"C:\Users\pc\Music\face_video\test_video\ytb1.mp4"

# Chạy dự đoán
label, confidence = predict_with_faces(video_path)
print(f"Predicted Label: {label} (Confidence: {confidence:.2f})")
