import cv2
import os

#Setting up the camera

#Folder

classes = int(input('classes: '))


def folder_scratch():
    archivos = os.listdir()
    numeros = []
    for archivo in archivos:
        nombre, extension = os.path.splitext(archivo)
        try:
            numero = int(nombre)
            numeros.append(numero)
        except ValueError:
            pass

    if numeros:
        numero_mas_alto = max(numeros)
        return numero_mas_alto + 1
    else:
        return 0    
    

#Capturing the dataset
def capture(frameNr):
    
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224) # insert the width at which the model was trained 

    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)# insert the height at which the model was trained (usually same as width) 

    cap.set(cv2.CAP_PROP_FPS, 30) # You can vary the frames based on the dataset len you want to achieve

    while (True):

        success, frame = cap.read()
        
        cv2.imshow('frame', frame)
        
        if success:
            cv2.imwrite(f'{frameNr}.jpg', frame)

        else:
            raise(RuntimeWarning)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frameNr = frameNr+1

    cap.release()
    cv2.destroyAllWindows()

def runtime():
    for i in range(classes):
        if not os.path.exists(os.path.join(os.getcwd(), str(i))):            
            os.makedirs(os.path.join(os.getcwd(), str(i)))
            os.chdir(os.path.join(os.getcwd(), str(i)))
            input()
            capture(frameNr = 0)
            os.chdir(os.path.dirname(os.getcwd()))
        else:
            os.chdir(os.path.join(os.getcwd(), str(i)))
            frame = folder_scratch()
            input()
            capture(frameNr = frame)
            os.chdir(os.path.dirname(os.getcwd()))
        


runtime()

input('After getting the dataset, you are able to delete this file')