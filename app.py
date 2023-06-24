import torch
import cv2
from utils import *
from models import *

#Define n_classes and path

path, n_classes = get_class_path()

#Define the model architecture and the transformations to the dataset. You have to choose the same model in every step

#transform, _ = Shufflenet(n_classes)
#transform, _ = Resnet34(n_classes)
#transform, _ = Resnet18(n_classes)
transform, _ = Resnet50(n_classes)


# Importing the model
model = torch.jit.load('model.pt')
model.eval()

#prediction function
def ai(input):
    a = model(transform(input).unsqueeze(0))
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



