import torch.nn as nn
import torch
import torchvision.transforms as tt
import cv2
from utils import *
from torchvision.models import shufflenet_v2_x0_5, ShuffleNet_V2_X0_5_Weights

n_classes = int(input('classes: '))

#Model class

model = shufflenet_v2_x0_5(weights=ShuffleNet_V2_X0_5_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad_ = False

model.fc = nn.Sequential(
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, n_classes)
)

class Model(ImageClassificationBase):
    def __init__(self):
        super().__init__()

        self.model = model

    def forward(self, xb):
        out = self.model(xb)
        return out


# Importing the model
model = Model()
model.load_state_dict(torch.load('model.pt', map_location=torch.device('cpu')))
model.eval()

# Normalize and transforms opencv to pytorch
mean=(0.485, 0.456, 0.406)
std=(0.229, 0.224, 0.225)
stats = (mean, std)

preprocess = tt.Compose([
    tt.ToTensor(),
    tt.Normalize(*stats)
])

#prediction function
def ai(input):
    a = model(preprocess(input).unsqueeze(0))
    return torch.argmax(a).item()

#Importing camera frames
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 256)
cap.set(cv2.CAP_PROP_FPS, 30)


while True:
    
    #Import frame
    _, frame = cap.read()
    
    #pasar a RGB
    input_= frame[:, :, [2, 1, 0]]    
    
    # Dibuja el mensaje en el fotograma
    cv2.putText(frame, f'{ai(input_)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    #Mostrar en pantalla
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()



