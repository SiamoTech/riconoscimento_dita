#importa le librerie
import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Inizializza la soluzione Hands
hands = mp_hands.Hands()

# Cattura il video dalla webcam
cap = cv2.VideoCapture(0)

# Imposta la risoluzione del video
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Dizionario per convertire gli indici delle dita in nomi di dita
finger_dict = {0: 'Pollice', 1: 'Indice', 2: 'Medio', 3: 'Anulare', 4: 'Mignolo'}

while cap.isOpened():
    # Leggi un frame dal video
    success, image = cap.read()
    if not success:
        break

    # Converti l'immagine in scala di grigi
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calcola la luminosità media dell'immagine
    brightness = np.mean(gray)

    # Adatta dinamicamente i parametri del codice in base alla luminosità dell'ambiente
    brightness_adjustment = 0
    if brightness < 50:
        # Ambiente poco illuminato
        image = cv2.add(image, np.ones_like(image) * 50)
        brightness_adjustment = 5
    elif brightness > 200:
        # Ambiente molto illuminato
        image = cv2.add(image, np.ones_like(image) * -50)
        brightness_adjustment = -5
    else:
        # Ambiente normalmente illuminato
        pass

    # Converti l'immagine in RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Rileva le mani nell'immagine
    results = hands.process(image)

    # Inizializza la variabile hand_distance_color con un valore predefinito
    hand_distance_color = (255, 0, 0)

    # Inizializza la variabile distance con un valore predefinito
    distance = 0.5

    # Disegna i risultati sulle mani rilevate
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Conta il numero di dita alzate e quali dita sono alzate
            count = 0
            raised_fingers = []
            if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x < hand_landmarks.landmark[2].x:
                count += 1
                raised_fingers.append(finger_dict[0])
            if hand_landmarks.landmark[8].y < hand_landmarks.landmark[7].y < hand_landmarks.landmark[6].y:
                count += 1
                raised_fingers.append(finger_dict[1])
            if hand_landmarks.landmark[12].y < hand_landmarks.landmark[11].y < hand_landmarks.landmark[10].y:
                count += 1
                raised_fingers.append(finger_dict[2])
            if hand_landmarks.landmark[16].y < hand_landmarks.landmark[15].y < hand_landmarks.landmark[14].y:
                count += 1
                raised_fingers.append(finger_dict[3])
            if hand_landmarks.landmark[20].y < hand_landmarks.landmark[19].y < hand_landmarks.landmark[18].y:
                count += 1
                raised_fingers.append(finger_dict[4])

            # Stampa il numero di dita alzate e quali dita sono alzate
            print("Numero di dita alzate:", count)
            print("Dita alzate:", ', '.join(raised_fingers))
            # Imposta il colore del testo e delle linee in base alla luminosità dell'ambiente
            text_color = (255, 255, 255)
            line_color_lum = (255, 255, 255)
            if brightness > 200:
                # Ambiente molto illuminato
                text_color = (0, 0, 0)
                line_color_lum = (0, 0, 0)
            elif brightness < 50:
                # Ambiente poco illuminato
                text_color = (255, 255, 255)
                line_color_lum = (255, 255, 255)

            # Visualizza il numero di dita alzate e quali dita sono alzate sull'immagine
            cv2.putText(image, f"Dita alzate: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
            cv2.putText(image, f"Dita: {', '.join(raised_fingers)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        text_color, 2)

            # Calcola la distanza della mano dalla webcam
            # Assegna la coordinata z del landmark del polso (indice 0) alla variabile wrist_z
            wrist_z = hand_landmarks.landmark[0].z
            # Assegna la coordinata z del landmark della punta del dito medio (indice 12) alla variabile middle_finger_tip_z
            middle_finger_tip_z = hand_landmarks.landmark[12].z
            # Calcola la distanza tra il polso e la punta del dito medio come il valore assoluto della differenza tra wrist_z e middle_finger_tip_z
            distance = abs(wrist_z - middle_finger_tip_z)

            # Imposta il colore del puntino sullo slider in base alla distanza della mano dalla webcam
            if distance > 0.05:
                # Mano troppo vicina
                hand_distance_color = (0, 0, 255)
            elif distance < 0.01:
                # Mano troppo lontana
                hand_distance_color = (0, 0, 255)
            else:
                # Distanza ottimale
                hand_distance_color = (255, 0, 0)
    else:
        # La mano non è stata rilevata o tracciata correttamente
        hand_distance_color = (255, 255, 255)

    # Disegna una linea sulla sinistra dell'interfaccia per indicare la luminosità dell'ambiente
    # Crea un'immagine vuota delle stesse dimensioni dell'immagine originale
    line_image = np.zeros_like(image)
    # Ottieni l'altezza e la larghezza dell'immagine
    height, width = line_image.shape[:2]
    # Definisci il punto di partenza e di arrivo della linea
    line_start = (20, int(height * 0.8))
    line_end = (20, int(height * 0.9))
    # Definisci il colore della linea
    line_color_lum = (255, 255, 255)
    # Disegna la linea sull'immagine
    cv2.line(line_image, line_start, line_end, line_color_lum, 3)
    # Calcola la posizione del cerchio sulla linea in base alla luminosità dell'ambiente
    circle_pos = (line_end[0], int(line_end[1] - (line_end[1] - line_start[1]) * (brightness / 255)))

    # Disegna il cerchio sulla linea
    cv2.circle(line_image, circle_pos, 10, (0, 0, 255), -1)
    # Definisci il colore del testo
    text_color_lum = (255, 255, 255)

    # Scrivi il testo sull'immagine per indicare il livello di luminosità dell'ambiente
    cv2.putText(line_image,
                f"{int(brightness / 51) + 1}",
                (circle_pos[0] - 5,
                 circle_pos[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                text_color_lum,
                2)

    # Disegna una linea sulla destra dell'interfaccia per indicare la distanza ottimale della mano dalla webcam
    distance_line_image = np.zeros_like(image)
    height, width = distance_line_image.shape[:2]
    distance_line_start = (width - 40, int(height * 0.8))
    distance_line_end = (width - 40, int(height * 0.9))
    line_color_dist = (255, 255, 255)
    cv2.line(distance_line_image,
             distance_line_start,
             distance_line_end,
             line_color_dist,
             3)
    # Calcola la posizione del cerchio sulla linea in base alla distanza della mano dalla webcam
    # La posizione del cerchio viene calcolata come una tupla di due elementi.
    # Il primo elemento è la coordinata x del punto finale della linea,
    # mentre il secondo elemento è la coordinata y del punto finale della linea meno la differenza tra le coordinate y del punto finale e del punto iniziale della linea
    # moltiplicata per la distanza della mano dalla webcam.
    circle_pos = (
        distance_line_end[0],
        int(distance_line_end[1] -
            (distance_line_end[1] -
             distance_line_start[1]) *
            distance))

    cv2.circle(distance_line_image,
               circle_pos,
               10,
               hand_distance_color,
               -1)

    text_color_dist = (255, 255, 255)
    if results.multi_hand_landmarks:
        cv2.putText(distance_line_image,
                    f"{int(distance * 5) + 1}",
                    (circle_pos[0] - 5,
                     circle_pos[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    text_color_dist,
                    2)
    else:
        # Scrivi "ND" sull'immagine per indicare che la distanza non è disponibile
        cv2.putText(distance_line_image,
                    "ND",
                    (circle_pos[0] - 5,
                     circle_pos[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    text_color_dist,
                    2)

    # Visualizza l'immagine con i risultati
    image = cv2.add(image, line_image)
    image = cv2.add(image, distance_line_image)
    cv2.imshow('Image', image)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
