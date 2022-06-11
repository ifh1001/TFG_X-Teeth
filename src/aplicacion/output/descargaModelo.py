# Archivo que permite descargar el modelo de la aplicación.
#
# Realizado por Ismael Franco Hernando.

# -- Imports --
import gdown

# URL de Google Drive que contiene el modelo de Detectron2 para poder
# descargarse.
url = "https://drive.google.com/uc?id=1-8-XLbs9SrmPS1cKqar2sy1-9b_rrfcq"

# Se descarga el modelo de la URL con el nombre correcto para que pueda ser
# usado por la aplicación.
gdown.download(url, output='./model_final.pth')
