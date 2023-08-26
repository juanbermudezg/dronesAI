import cv2
import numpy as np
import tensorflow as tf
from djitellopy import Tello
import random
import time

# Conectar al dron Tello
tello = Tello()
tello.connect()
tello.streamon()

# Cargar el nuevo modelo SavedModel
ruta_al_modelo = 'python/content/carpeta_salida/modelo_estructuras/1'
modelo = tf.saved_model.load(ruta_al_modelo)

# Clases del modelo
clases = ['Circulo', 'Cuadrado', 'Rectangulo']
umbral_confianza = 0.8

#tello.move_down(20)


while True:
    print("distance ",tello.get_distance_tof())
    print("batery: ", tello.get_battery())
    imgdrone = tello.get_frame_read().frame
    frame= cv2.cvtColor(imgdrone, cv2.COLOR_BGR2RGB)

    # Realizar la detección de objetos
    resized_frame = cv2.resize(frame, (224, 224))
    resized_frame2 = cv2.resize(frame, (224, 224)).astype('float32')
    normalized_frame = resized_frame2 / 255.0

    input_data = {'keras_layer_input': normalized_frame[np.newaxis, ...]}
    predictions = modelo.signatures["serving_default"](**input_data)

    output_data = predictions['dense']

    predicted_class = np.argmax(output_data, axis=-1)[0]
    confidence = output_data[0][predicted_class]
    if cv2.waitKey(1) & 0xFF == ord('t'):
        tello.takeoff()
        tello.set_speed(100)
        tello.move_up(70) 

    if confidence > umbral_confianza:
        predicted_label = clases[predicted_class]
        cv2.putText(frame, f'Objeto: {predicted_label} ({confidence:.2f})', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        # Realizar la detección de colores dentro del objeto
        hsv = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2HSV)

        # Definir rangos de colores en HSV
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([120, 255, 255])
        
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])
        
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)

        mask_red = cv2.bitwise_or(mask_red1, mask_red2)

        res_yellow = cv2.bitwise_and(resized_frame, resized_frame, mask=mask_yellow)
        res_blue = cv2.bitwise_and(resized_frame, resized_frame, mask=mask_blue)
        res_green = cv2.bitwise_and(resized_frame, resized_frame, mask=mask_green)
        res_red = cv2.bitwise_and(resized_frame, resized_frame, mask=mask_red)

        # Determinar el color predominante dentro del objeto
        color_counts = {
            "Amarillo": np.sum(mask_yellow > 0),
            "Azul": np.sum(mask_blue > 0),
            "Verde": np.sum(mask_green > 0),
            "Rojo": np.sum(mask_red > 0),
        }
        print(np.sum(mask_red > 0))
        
        predominant_color = max(color_counts, key=color_counts.get)

        # Mostrar el color predominante dentro del objeto
        cv2.putText(frame, f'Color Predominante: {predominant_color}', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        
        
            #tello.move_forward(100)
        if predicted_label == 'Circulo' and  np.sum(mask_red > 0) >4000:
            try:
                cv2.imwrite('frame_capturado.jpg', frame) # Prueba
                tello.land()
                break
            except:
                pass
        elif ((predicted_label=='Cuadrado' and predominant_color!="Verde") or predicted_label=='Rectangulo') and predominant_color=="Rojo":
            try:
                tello.move_down(80)
                tello.move_forward(200)
                tello.move_up(80)
                #pass
            except:
                pass
       
            
            
    # Mostrar los resultados en la ventana
    cv2.imshow('Dron Video', frame)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        tello.land()
        break
    try:
        tello.move_forward(150)
        time.sleep(3)
    except:
        pass
    
# Detener el flujo de video y desconectar el dron
tello.streamoff()
cv2.destroyAllWindows()