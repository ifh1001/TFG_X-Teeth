# Archivo que contiene todas las funciones para que pueda funcionar la 
# aplicación X-Teeth.
#
#
# Realizado por Ismael Franco Hernando.

# -- Imports --
import io
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from PIL import Image
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer, ColorMode
from skimage.morphology import thin
from scipy.spatial import distance


# -- Clases --
class Predictor(object):
    __instancia = None
    
    def __new__(cls):
        if Predictor.__instancia is None:
            Predictor.__instancia = obtiene_predictor()
        return Predictor.__instancia 

# -- Funciones --

# Método encargado de controlar todo el proceso para calcular y  mostrar la
# longitud del diente.
#
# Parámetros:
#  - uploader: elemento que contendrá la imagen con la que se quiere trabajar.
def aplicacion(uploader):
    predictor = Predictor()
    for name, file_info in uploader.value.items():
        print(name)
        imagen_bytes = Image.open(io.BytesIO(file_info['content']))
        imagen = cv2.cvtColor(np.array(imagen_bytes), cv2.COLOR_RGB2BGR)
        
        puntoA, puntoB, distancia, im_predicc, puntos = calcula_distancia(imagen, predictor)
        muestra_imagenes(imagen, im_predicc, puntoA, puntoB, puntos)
        muestra_distancia(distancia)


# Método encargado de cargar y devolver el predictor.
#
# Return:
#  - DefaultPredictor(cfg): predictor con la configuración establecida.
def obtiene_predictor():
    cfg = obtiene_configuracion()

    return DefaultPredictor(cfg)


# Método encargado de cargar y devolver la configuración.
#
# Return:
#  - cfg: configuración establecida para el predictor.
def obtiene_configuracion():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 2500
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.TEST.DTECTIONS_PER_IMAGE = 2
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.50

    return cfg


# Método encargado de calcular la longitud del diente.
#
# Parámetros:
#  - imagen: imagen con la que se quiere trabajar.
#  - predictor: predictor que permita segmentar la imagen.
# Return:
#  - PuntoA: punto de la forma (y,x) de la parte superior del diente.
#  - PuntoB: punto de la forma (y,x) de la parte inferior del diente.
#  - distancia: distancia entre el puntoA y el puntoB.
#  - im_predicc: imagen con la predicción del Detectron2.
#  - puntos: puntos de intersección entre el thin y las rectas perpendiculares.
def calcula_distancia(imagen, predictor):
    predicciones = predictor(imagen)
    resolucion_y = 41 / imagen.shape[0]
    resolucion_x = 31 / imagen.shape[1]
    resolucion = 41 / imagen.shape[0]

    diente, nervio = ordena_predicciones(predicciones)

    nervio_thin = thin(nervio)
    puntos_mitad = np.where(nervio_thin == True)
    puntos_diente = np.where(diente == True)
    punto1 = (puntos_mitad[1][0], puntos_mitad[0][0])
    punto2 = (puntos_mitad[1][-1], puntos_mitad[0][-1])

    # Cálculo punto de la parte superior del diente
    filas_diente_min = np.where(puntos_diente[0] == np.min(puntos_diente[0]))
    y = puntos_diente[0][filas_diente_min[0][0]]
    y, x = punto_correcto(diente, punto1, punto2, y)
    puntoA = y, x

    # Cálculo punto de la parte inferior del diente
    filas_diente_max = np.where(puntos_diente[0] == np.max(puntos_diente[0]))
    y = puntos_diente[0][filas_diente_max[0][0]]
    y, x = punto_correcto(diente, punto1, punto2, y, superior=False)
    puntoB = y, x

    # Cálculo de la distancia entre los puntos
    distancia, puntos = distancia_entre_partes(nervio_thin, puntos_mitad, puntoA, puntoB)
    
    #Preparamos imagen con las predicciones
    v = Visualizer(imagen[:, :, ::-1],
                   scale=1,
                   instance_mode=ColorMode.IMAGE_BW)
    
    im_predicc = v.draw_instance_predictions(predicciones["instances"].to("cpu"))
    return puntoA, puntoB, distancia * resolucion, im_predicc, puntos


# Método encargado de calcular la longitud del diente, junto con los distintos
# puntos del nervio sobre los que se cálcula la distancia.
#
# Parámetros:
#  - nervio_thin: máscara del nervio obtenida con el thin.
#  - puntos_mitad: puntos de la máscara thin.
#  - puntoA: punto de la forma (y,x) de la parte superior del diente.
#  - puntoB: punto de la forma (y,x) de la parte inferior del diente.
# Return:
#  - distancia: longitud total del diente.
#  - puntos: puntos de intersección entre el thin y las rectas perpendiculares.
def distancia_entre_partes(nervio_thin, puntos_mitad, puntoA, puntoB):
    punto_inicio = (puntos_mitad[0][0], puntos_mitad[1][0])
    punto_fin = (puntos_mitad[0][-1], puntos_mitad[1][-1])
    
    # Cálculo pendiente
    pendiente = (puntoB[0] - puntoA[0]) / (puntoB[1] - puntoA[1]) 
    
    # Obtenciíon recta perpendicular
    pend_perpendicular = - 1 / pendiente    
    constante = -(pend_perpendicular * punto_inicio[1]) + punto_inicio[0]    
    
    #Puntos de intersección entre recta perpendicular y el thin del diente
    puntos = dibuja_perpendiculares(pend_perpendicular, constante, punto_inicio, punto_fin, puntos_mitad)
    
    # Cálculo distancia
    puntos.append(punto_fin)
    distancia = distancia_entre_puntos(puntos)    
    distancia = distancia + distance.euclidean((punto_inicio[1], punto_inicio[0]), (puntoA[1], puntoA[0]))
    distancia = distancia + distance.euclidean((punto_fin[1], punto_fin[0]), (puntoB[1], puntoB[0]))    
    
    return distancia, puntos


# Método encargado de calcular la longitud entre cada par de puntos.
#
# Parámetros:
#  - puntos: puntos de intersección entre el thin del nervio y las rectas
#            perpendiculares.
# Return:
#  - distancia: longitud total entre los puntos.
def distancia_entre_puntos(puntos):
    distancia = 0
    for i in range(len(puntos)-1):
        punto1 = puntos[i]
        punto2 = puntos[i+1]
        distancia = distancia + distance.euclidean((punto1[1], punto1[0]), (punto2[1], punto2[0]))
        
    return distancia


# Método encargado de obtener los puntos de corte entre las rectas
# perpendiculares y la máscara thin.
#
# Parámetros:
#  - pend_perpendicular: pendiente de la recta perpendicular.
#  - constante: constante de la recta perpendicular que pasa por el primer
#               punto.
#  - punto_inicio: primer punto de la máscara thin.
#  - punto_fin: último punto de la máscara thin.
#  - puntos_thin: todos los puntos de las máscara thin del nervio.
# Return:
#  - lista_puntos: lista con todos los puntos de intersección.
def dibuja_perpendiculares(pend_perpendicular, constante, punto_inicio, punto_fin, puntos_thin):
    divisiones = 20    
    distancia = (punto_fin[0] - punto_inicio[0]) / divisiones
    lista_puntos = []    
    
    for i in range(divisiones):    
        lista_puntos.append(comprueba_puntos(puntos_thin, pend_perpendicular, constante + distancia*i))
    
    return lista_puntos


# Método encargado de obtener el punto de intersección entre una recta y la
# máscara thin.
#
# Parámetros:
#  - puntos_thin: todos los puntos de las máscara thin del nervio.
#  - pend_perpendicular: pendiente de la recta perpendicular.
#  - constante: constante de la recta perpendicular que pasa por el primer
#               punto.
# Return:
#  - punto: punto de la forma (y,x) de la intersección.
def comprueba_puntos(puntos_thin, pend_perpendicular, constante):    
    for i in range(len(puntos_thin[1])):        
        y = pend_perpendicular*puntos_thin[1][i] + constante
        if int(y) == puntos_thin[0][i]:            
            return (int(y), puntos_thin[1][i])
            
        if int(y)+1 == puntos_thin[0][i]:            
            return (int(y)+1, puntos_thin[1][i])
        

# Método encargado de ordenar las predicciones.
#
# Parámetros:
#  - predicciones: contendrá las predicciones obtenidas del predictor.
# Return:
#  - diente: máscara correspondiente a la predicción del diente.
#  - nervio: máscara correspondiente a la predicción del nervio.
def ordena_predicciones(predicciones):
    diente = predicciones['instances'].pred_masks.to("cpu").numpy()[0]
    nervio = predicciones['instances'].pred_masks.to("cpu").numpy()[1]

    valor_d, contador_d = np.unique(diente, return_counts=True)
    valor_n, contador_n = np.unique(nervio, return_counts=True)
    if valor_d[1] == True:
        posic_d = 1
    else:
        posic_d = 0

    if valor_n[1] == True:
        posic_n = 1
    else:
        posic_n = 0

    if contador_d[posic_d] < contador_n[posic_n]:
        diente, nervio = nervio, diente

    return diente, nervio


# Método encargado de calcular la ecuación de la recta entre dos puntos, junto
# con la abscisa x a través de la ordenada y.
#
# Parámetros:
#  - punto1: primer punto del que se quiere calcular la recta.
#  - punto2: segundo punto del que se quiere calcular la recta.
#  - y: ordenada y, para calcular la abscisa x.
# Return:
#  - x: abscisa resultante.
def calcula_punto_recta(punto1, punto2, y):
    return (y - punto1[1]) / (punto2[1] - punto1[1]) * (punto2[0] - punto1[0]) + punto1[0]


# Método encargado de obtener el punto donde se corta la recta con el borde del
# diente.
#
# Parámetros:
#  - diente: máscara del diente predicho.
#  - punto1: primer punto del que se quiere calcular la recta.
#  - punto2: segundo punto del que se quiere calcular la recta.
#  - y: ordenada y, para calcular la abscisa x.
#  - superior: valor booleano para identificar si se quire calcular el punto
#              superior o inferior.
# Return:
#  - y,x: punto correspondiente al corte entre la recta y el borde del diente.
def punto_correcto(diente, punto1, punto2, y, superior=True):
    x = calcula_punto_recta(punto1, punto2, y)

    while not diente[y][int(x)]:
        if superior:
            y = y + 1
        else:
            y = y - 1

        x = calcula_punto_recta(punto1, punto2, y)    

    return y, int(x)


# Método encargado de mostrar la imagen original, junto con la imagen y la
# recta con la que se ha calculado la longitud del diente.
#
# Parámetros:
#  - imagen: imagen con la que se quiere trabajar.
#  - im_predicc: imagen con la predicción del Detectron2.
#  - puntoA: primer punto de la recta.
#  - puntoB: segundo punto de la recta.
#  - puntos: puntos de intersección entre el thin del nervio y las rectas
#            perpendiculares. 
def muestra_imagenes(imagen, im_predicc, puntoA, puntoB, puntos):
    fig = plt.figure()
    fig.set_size_inches(14, 22)
    fig.add_subplot(1, 3, 1)
    plt.title("Radiografía")
    plt.imshow(imagen)
    plt.axis('off')
    
    fig.add_subplot(1, 3, 2)
    plt.title("Segmentación")
    plt.imshow(im_predicc.get_image()[:, :, ::-1])
    plt.axis('off')
    
    fig.add_subplot(1, 3, 3)
    plt.title("Radiografía con Recta")
    plt.imshow(imagen)
    plt.plot([puntoA[1], puntos[0][1]], [puntoA[0], puntos[0][0]], color='r')
    plt.plot([puntoB[1], puntos[-1][1]], [puntoB[0], puntos[-1][0]], color='r')
    for i in range(len(puntos) - 1):
        punto1 = puntos[i]
        punto2 = puntos[i+1]
        plt.plot([punto1[1], punto2[1]], [punto1[0], punto2[0]], color='r')  
    plt.axis('off')

    plt.show()


# Método encargado de mostrar un widget con la distancia calculada.
#
# Parámetros:
#  - distancia: distancia que se quiere mostrar.
def muestra_distancia(distancia):
    distancia = round(distancia, 1)
    w = widgets.Text(value='La longitud del diente es: ' + str(distancia) + 'mm', disabled=True)
    display(w)
