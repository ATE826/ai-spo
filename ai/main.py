from ultralytics import YOLO
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import scipy.io.wavfile as wav
import cv2
import numpy as np

# load yolo model
modelYolo = YOLO('../runs/detect/train8/weights/best.pt')
feature_sequence = []

# image
image_path = 'ai\\test.jpg' # Path to test image to test YOLO on
image = cv2.imread(image_path)
image_height, image_width, _ = image.shape

# run model on image
results = modelYolo(image_path)
for result in results:
    boxes = result.boxes  # bounding boxes
    for box in boxes:
        x_center, y_center, width, height = box.xywhn[0].tolist()  # normalized coordinates
        confidence = box.conf.item()  # confidence score
        class_id = box.cls.item()  # class id
        x1 = int((x_center - width / 2) * image_width)
        y1 = int((y_center - height / 2) * image_height)
        x2 = int((x_center + width / 2) * image_width)
        y2 = int((y_center + height / 2) * image_height)
        star_region = image[y1:y2, x1:x2]
        brightness = np.mean(star_region) / 255.0  # normalize [0, 1]

        # make feature vector
        feature_vector = [x_center, y_center, width, height, brightness]
        feature_sequence.append(feature_vector)
        print(f"Star detected at: ({x_center}, {y_center}), size: ({width}, {height}), confidence: {confidence}")

feature_sequence = np.array(feature_sequence)

# max_length = 10  # Maximum number of stars per image
# padded_sequence = np.zeros((max_length, 5))  # 5 features per star
# padded_sequence[:len(feature_sequence)] = feature_sequence

# sound dataset
# class StarSoundDataset(Dataset):
#     def __init__(self, feature_sequences, sound_data):
#         self.feature_sequences = feature_sequences  # List of feature sequences
#         self.sound_data = sound_data  # List of corresponding sound data

#     def __len__(self):
#         return len(self.feature_sequences)

#     def __getitem__(self, idx):
#         features = torch.tensor(self.feature_sequences[idx], dtype=torch.float32)
#         sound = torch.tensor(self.sound_data[idx], dtype=torch.float32)
#         return features, sound

# dataset = StarSoundDataset(feature_sequence, [])   
# batch_size = 4
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# # RNN
# class SoundRNN(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(SoundRNN, self).__init__()
#         self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         out, _ = self.rnn(x)  # output
#         out = self.fc(out)  # fully connected layer
#         return out

# # define model params
# input_size = 4  # number of features (x_center, y_center, brightness, size)
# hidden_size = 128  # hidden units
# output_size = 128  # output features (sound waveform/midi params)

# # init model
# modelRnn = SoundRNN(input_size, hidden_size, output_size)
# criterion = nn.MSELoss()  # mean squared error
# optimizer = optim.Adam(modelRnn.parameters(), lr=0.001)

# # RNN training
# for epoch in range(100):  # epochs
#     for inputs, targets in dataloader:
#         optimizer.zero_grad()
#         outputs = modelRnn(inputs)
#         loss = criterion(outputs, targets)
#         loss.backward()
#         optimizer.step()
#     print(f"epoch {epoch+1}, loss: {loss.item()}")

# new_sequence = torch.tensor([], dtype=torch.float32) # FILL
# with torch.no_grad():
#     sound_output = modelRnn(new_sequence)

# wav.write('rnn\\output.wav', rate=44100, data=sound_output)