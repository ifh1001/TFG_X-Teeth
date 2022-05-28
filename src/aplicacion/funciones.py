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
from skimage.morphology import thin


# -- Funciones --

# Método encargado de controlar todo el proceso para calcular y  mostrar la
# longitud del diente.
#
# Parámetros:
#  - uploader: elemento que contendrá la imagen con la que se quiere trabajar.
def aplicacion(uploader):
    imagen = carga_imagen(uploader)
    predictor = obtiene_predictor()
    puntoA, puntoB, distancia = calcula_distancia(imagen, predictor)
    muestra_imagenes(imagen, puntoA, puntoB)
    muestra_distancia(distancia)


# Método encargado de transformar la imágen cargado en cadena de bytes a forma
# matricial.
#
# Parámetros:
#  - uploader: elemento que contendrá la imagen en cadena de bytes.
# Return:
#  - img: imagen cargada en forma ed matriz.
def carga_imagen(uploader):
    for name, file_info in uploader.value.items():
        imagen_bytes = Image.open(io.BytesIO(file_info['content']))
        img = cv2.cvtColor(np.array(imagen_bytes), cv2.COLOR_RGB2BGR)

    return img


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
def calcula_distancia(imagen, predictor):
    predicciones = predictor(imagen)
    resolucion_y = 41 / imagen.shape[0]
    resolucion_x = 31 / imagen.shape[1]

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
    xx = ((puntoA[1] - puntoB[1]) * resolucion_x) ** 2
    yy = ((puntoA[0] - puntoB[0]) * resolucion_y) ** 2
    distancia = (xx + yy) ** 0.5

    return puntoA, puntoB, distancia


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
#  - puntoA: primer punto de la recta.
#  - puntoB: segundo punto de la recta.
def muestra_imagenes(imagen, puntoA, puntoB):
    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.title("Radiografía")
    plt.imshow(imagen)
    plt.axis('off')

    fig.add_subplot(1, 2, 2)
    plt.title("Radiografía con Recta")
    plt.imshow(imagen)
    plt.plot([puntoA[1], puntoB[1]], [puntoA[0], puntoB[0]], color='r')
    plt.axis('off')

    plt.show()


# Método encargado de mostrar un widget con la distancia calculada.
#
# Parámetros:
#  - distancia: distancia que se quiere mostrar.
def muestra_distancia(distancia):
    distancia = round(distancia, 1)
    w = widgets.Text(value='La distancia es: ' + str(distancia) + 'mm', disabled=True)
    display(w)
