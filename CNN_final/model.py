import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from CNN.MixtureMasking import MixtureMasking

# Location of the correct path file
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'CNN', 'mixturemasking', 'violin_viola_cnn_mixturemasking_epoch_47.pth'
)

class ViolinViolaCNN(nn.Module):
    def __init__(self):
        super(ViolinViolaCNN, self).__init__()

        self.mixturemasking1 = MixtureMasking(freq_param=8, time_param=35, activate_prob=0.6)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=40, stride=(2, 1), kernel_size=(5, 1))   # output: (40, 62, 517)
        self.mixturemasking2 = MixtureMasking(freq_param=4, time_param=35, activate_prob=0.6)
        self.conv2 = nn.Conv2d(in_channels=40, out_channels=32, kernel_size=(3, 2))  # output: (32, 60, 516)
        self.mixturemasking3 = MixtureMasking(freq_param=4, time_param=35, activate_prob=0.6)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1)  # output: (16, 60, 516)

        self.pool = nn.AdaptiveAvgPool2d((16, 16))  # reduce to fixed-size for Dense

        self.fc1 = nn.Linear(16 * 16 * 16, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 12)
        self.out = nn.Linear(12, 2)  # binary classification: violin vs viola

    def forward(self, x, labels):
        x = self.mixturemasking1(x, labels)
        x = F.relu(self.conv1(x))  # -> (40, 62, 517)
        x = self.mixturemasking2(x, labels)
        x = F.relu(self.conv2(x))  # -> (32, 60, 516)
        x = self.mixturemasking3(x, labels)
        x = F.relu(self.conv3(x))  # -> (16, 60, 516)
        x = self.pool(x)           # -> (16, 16, 16)
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))    # -> (64)
        x = F.relu(self.fc2(x))    # -> (16)
        x = F.relu(self.fc3(x))    # -> (12)
        x = self.out(x)            # -> (2)
        return x

class ViolinViolaCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(ViolinViolaCrossEntropyLoss, self).__init__()
        self.register_buffer("class_weights", torch.tensor([1.0, 1.2]))

    def forward(self, outputs, labels):
        if self.class_weights is not None:
            class_weights = self.class_weights.to(outputs.device)
            return F.cross_entropy(outputs, labels, weight=class_weights)
        return F.cross_entropy(outputs, labels)
    
class ViolinViolaDataset(Dataset):
    def __init__(self, root_dir, label_map = None, transform=None):
        self.root_dir = root_dir
        self.files = [f for f in os.listdir(root_dir) if f.endswith('.npy')]
        self.label_map = label_map or {'violin': 0, 'viola': 1}
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        filepath = os.path.join(self.root_dir, filename)
        mel = np.load(filepath)
        spec = torch.from_numpy(mel)
        spec = spec.unsqueeze(0)  # Add channel dimension
        label = self.label_map[filename.split('_')[1]] if self.label_map else None
        if self.transform:
            spec = self.transform(spec)
        return spec, label
        
transform = transforms.Compose([
    transforms.Resize((128, 517)),  # Resize to a fixed size
    transforms.Normalize([0.5], [0.5])  # Normalize the image
])

    
def load_model():
    model = ViolinViolaCNN()
    # Best model state_dict
    state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    return model

def predict_spectrogram(model, audio_array):
    """
    Predict the class of the audio sample using the trained model.
    Parameters:
    model (ViolinViolaCNN): The trained model.
    audio_array (np.ndarray): The mel spectrogram as a numpy array.
    Returns:
    np.ndarray: Predicted probabilities for each class.
    """
    model.eval()
    with torch.no_grad():
        audio_tensor = torch.tensor(audio_array).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        input_tensor = transform(audio_tensor.float())
        output = model(input_tensor, None)
        return output.softmax(dim=1).cpu().numpy()

def make_prediction(model, audio_list):
    """
    Predict the class of the audio samples using the trained model.
    Parameters:
    model (ViolinViolaCNN): The trained model.
    audio_list (list): List of mel spectrograms as numpy arrays.
    Returns:
    List: Predicted probabilities for each class.
    """
    probs_list = [predict_spectrogram(model, audio_array) for audio_array in audio_list]
    probs = np.mean(probs_list, axis=0).flatten()
    probs = probs.tolist()  # Convert to list for JSON serialization
    probs_percent = [round(p * 100, 2) for p in probs]  # Convert to percentage
    return probs_percent  # Return percentage list