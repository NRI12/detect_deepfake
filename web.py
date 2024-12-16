import cv2
import torch
import numpy as np
import gradio as gr
import face_recognition
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision import models
from torch import nn
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

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

def extract_faces_from_video(video_path, output_size=(112, 112), num_frames=16):
    """
    Extract faces from video and return frames containing only faces.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    face_frames = []
    face_locations_list = []
    
    success, frame = cap.read()
    
    while success and len(frames) < num_frames:
        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Find faces in frame
        face_locations = face_recognition.face_locations(rgb_frame)
        
        for (top, right, bottom, left) in face_locations:
            # Crop face
            face = rgb_frame[top:bottom, left:right]
            
            # Resize face to standard size
            face = cv2.resize(face, output_size)
            
            frames.append(frame)  # Original frame for visualization
            face_frames.append(face)
            face_locations_list.append((top, right, bottom, left))
            break  # Only take first face per frame
        
        success, frame = cap.read()
    
    cap.release()
    
    # Pad if fewer frames than required
    while len(face_frames) < num_frames:
        face_frames.append(np.zeros((output_size[0], output_size[1], 3), dtype=np.uint8))
        frames.append(np.zeros((output_size[0], output_size[1], 3), dtype=np.uint8))
        face_locations_list.append((0,0,0,0))
    
    # Apply transform
    transform = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    processed_frames = torch.stack([transform(frame) for frame in face_frames]).unsqueeze(0)
    
    return processed_frames, frames, face_locations_list

def predict_with_faces(video_path):
    """
    Use model to predict video label after extracting faces.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = DeepfakeModel().to(device)
    model.load_state_dict(torch.load('deepfake_model.pth', map_location=device))
    model.eval()
    
    # Extract faces
    frames, original_frames, face_locations = extract_faces_from_video(video_path)
    frames = frames.to(device)
    
    # Predict
    with torch.no_grad():
        logits = model(frames)
        probabilities = torch.softmax(logits, dim=1)
        predicted_label = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_label].item()
    
    label_name = "Real" if predicted_label == 1 else "Fake"
    
    return label_name, confidence, original_frames, face_locations

def create_visualization(original_frames, face_locations):
    """
    Create a detailed visualization of frames with face detections.
    """
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    fig.suptitle('Detected Frames with Face Locations', fontsize=16)
    
    for i, (frame, (top, right, bottom, left)) in enumerate(zip(original_frames, face_locations)):
        ax = axes.ravel()[i]
        
        # Convert BGR to RGB if needed
        if frame.ndim == 3 and frame.shape[2] == 3:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = frame
        
        ax.imshow(frame_rgb)
        
        # Draw rectangle for face location
        if left != 0 or top != 0 or right != 0 or bottom != 0:
            rect = plt.Rectangle((left, top), right-left, bottom-top, 
                                 fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
        
        ax.axis('off')
    
    plt.tight_layout()
    return fig

def gradio_predict(video):
    """
    Gradio interface prediction function.
    """
    try:
        label, confidence, original_frames, face_locations = predict_with_faces(video)
        
        # Create visualization with face locations
        result_image = create_visualization(original_frames, face_locations)
        
        return label, confidence, result_image
    except Exception as e:
        return f"Error: {str(e)}", 0, None

# Create Gradio Interface with improved layout
with gr.Blocks() as iface:
    # Title and Description
    gr.Markdown("# 🕵️ Deepfake Detection System")
    gr.Markdown("Upload a video to detect whether it's real or manipulated.")
    
    with gr.Row():
        # Video Input
        video_input = gr.Video(label="Upload Video")
        
        # Results Column
        with gr.Column():
            # Prediction Results
            label_output = gr.Textbox(label="Prediction", interactive=False)
            confidence_output = gr.Number(label="Confidence", interactive=False)
    
    # Detection Button
    detect_btn = gr.Button("Detect Deepfake")
    
    # Visualization Output
    face_visualization = gr.Plot(label="Detected Faces")
    
    # Prediction Logic
    detect_btn.click(
        fn=gradio_predict, 
        inputs=video_input, 
        outputs=[label_output, confidence_output, face_visualization]
    )

# Launch the interface
iface.launch(debug=True)