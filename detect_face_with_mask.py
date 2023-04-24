# importe a biblioteca opencv
import cv2
import numpy as np

# biblioteca de aprendizado
import tensorflow as tf

model = tf.keras.models.load_model('keras_model.h5')
  

# defina um objeto de captura de vídeo
vid = cv2.VideoCapture(0)
  
while(True):
      
    # Capture o vídeo quadro a quadro
    ret, frame = vid.read()

    img = cv2.resizr(frame, (224,224))

    teste_image =  np.array(img, dtype=np.float32)
  
    teste_image = np.expand_dims(teste_image, axis=0)
    # Normalizar a imagem
    normalised_image = teste_image/255

    prediciton = model.predit(normalised_image)
    print('Previsão:', prediciton)

    # Exiba o quadro resultante
    cv2.imshow('quadro', frame)
      
    # Saia da tela com a barra de espaço
    key = cv2.waitKey(1)
    
    if key == 32:
        break
  
# Após o loop, libere o objeto capturado
vid.release()

# Destrua todas as janelas
cv2.destroyAllWindows()