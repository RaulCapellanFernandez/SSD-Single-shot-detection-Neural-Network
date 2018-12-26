#Librerias
import torch #Herramienta para crear redes neuronales dynamic grpahs
from torch.autograd import Variable #Convertir los tensores en variables
import cv2 #Open CV para dibujar los rectangulos
from data import BaseTransform, VOC_CLASSES as labelmap #Las carpetas del fichero data y les da formato a las imagenes
#VO_CLASSES los diferentes objetos que vamos a detectar
from ssd import build_ssd #Librerias del algoritmo
import imageio #Procesar las imagenes del video


#Funcion de deteccion parecida con Open CV
def detect(frame, neuralNet, transformacion):
    #altura y anchura
    #El vector frame.shape tiene de parametros altura anchura y numero de colores (1 BN 4 RGB)
    altura, anchura = frame.shape[:2]#Coge solo altura y anchura
    #Frame transformado
    frame_t = transformacion(frame)[0]
    x= torch.from_numpy(frame_t).permute(2,0,1)#Pasa la imagen transformada a los tensores de la red
    #Cambia el orden de los colores para que correspondan con torch
    x = Variable(x.unsqueeze(0))
    y = neuralNet(x)#le pasa la imagen a la red pre entrenada
    detections = y.data#Nuevo tensor le pasa la imagen y recibe la salida
    scalar = torch.Tensor([anchura, altura, anchura, altura])#Nuevo tensor para la altura y anchura de la imagen 
    #Coge el punto de arriba a al izqiuerda y abajo a la derecha del rectangulo
    
    #detections = [grupo, nuemero de tipos de objetos a detectar, numero de aparicion(1 perro 2 perros...), (score, x0,y0,x1,y1)]
    #score porcentaje para que sea deectado tiene que tener mas de un 60 porciento de acirto y sus coordenadas
    for i in range(detections.size(1)):
        j=0#Accurancy j , i el tipo de objeto
        while detections[0,i,j,0]>=0.6:
            punto = (detections[0,i,j,1:] * scalar).numpy()#Coge el rectanguo donde sale la imagen 1: coge las coordenadas completas
            #convertir en Numpy porque open Cv no trabaja con Torch
            cv2.rectangle(frame, (int (punto[0]), int (punto[1])), (int (punto[2]), int (punto[3])), (255, 0, 0), 2)#Coordenadas x0, y0 , x1, y1
            #Color rectangulo 255, 0, 0 y anchura del rectangulo 2
            cv2.putText(frame, labelmap[i-1], (int (punto[0]), int (punto[1])),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, (255,2555,255),2,cv2.LINE_AA)
            j+=1#Para mas objetos
    return frame

#Neural network object
net  = build_ssd('test')
#Cargar los pesos del modelo preentrenado
net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth', map_location = lambda storage, loc: storage))

#Crear la transformacion de la imagen para que funcione con la red neuronal
transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0))
# net.size tamaño aceptado por la red neuronal
#Escalar la imagen a las dimensiones deseadas para el color 

#Abrir el video
reader = imageio.get_reader('funny_dog.mp4')
#Frecuancia del Frame fps
fps= reader.get_meta_data()['fps']
#Crea el video que se va a mostrar como resultado final
writer = imageio.get_writer('funny_dog_output.mp4', fps = fps)
#Les pone los rectangulos al video original
for i, frame in enumerate(reader):
    #Sobre escribe la variable frame para incluirla conlos rectangulos
    frame = detect(frame, net.eval(), transform)
    #Lo incluye el frame al nuevo video
    writer.append_data(frame)
    print(i)
#Cierra el archivo de salida    
writer.close()