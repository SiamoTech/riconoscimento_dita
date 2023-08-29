import cv2
import mediapipe as mp
import numpy as np
import PySimpleGUI as sg

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Inizializza la soluzione Hands
hands = mp_hands.Hands()

# Crea una colonna di sinistra con i widget di testo e le barre di avanzamento
left_column = [
    [sg.Text("Dita alzate:", key='-FINGER_COUNT-')],
    [sg.Text("Dita:", key='-RAISED_FINGERS-')],
    [sg.Text("Movimento:", key='-MOVEMENT-')],
    [sg.Text("Ultimi 5 movimenti:", key='-LAST_MOVEMENTS-')],
    [sg.Text("Luminosità ambientale:")],
    [sg.ProgressBar(100, orientation='h', size=(20, 20), key='-BRIGHTNESS_BAR-')],
    [sg.Text("Distanza ottimale:")],
    [sg.ProgressBar(100, orientation='h', size=(20, 20), key='-DISTANCE_BAR-')],
    [sg.Text("", key='-HAND_DETECTION_STATUS-')]
]

# Crea una colonna di destra con un widget di immagine
right_column = [
    [sg.Image(key='-RIGHT_IMAGE-')]
]

# Crea il layout della finestra con due colonne
layout = [
    [sg.Column(left_column), sg.Column(right_column)],
    [sg.Button('Esci')]
]

# Crea la finestra
window = sg.Window("Gesture Hands", layout, finalize=True)

# Cattura il video dalla webcam
cap = cv2.VideoCapture(0)

# Imposta la risoluzione del video
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Dizionario per convertire gli indici delle dita in nomi di dita
finger_dict = {0: 'Pollice', 1: 'Indice', 2: 'Medio', 3: 'Anulare', 4: 'Mignolo'}

# Inizializza una variabile per memorizzare la posizione della mano nel frame precedente
prev_hand_pos = None

# Soglia di distanza per il rilevamento dei movimenti
movement_threshold = 0.01

# Inizializza una lista per memorizzare gli ultimi movimenti rilevati
last_movements = []

while True:
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

    # Inizializza la variabile hand_movement con un valore predefinito
    hand_movement = 'Nessun movimento'

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

            # Aggiorna il testo nella colonna di sinistra
            window['-FINGER_COUNT-'].update(value=f"Dita alzate: {count}")
            window['-RAISED_FINGERS-'].update(value=f"Dita: {', '.join(raised_fingers)}")

            # Calcola la posizione della mano utilizzando le coordinate x e y del landmark del polso (indice 0)
            hand_pos = (hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y)

            # Confronta la posizione della mano con la posizione precedente per determinare il movimento della mano.
            if prev_hand_pos is not None:
                if hand_pos[0] - prev_hand_pos[0] > movement_threshold:
                    hand_movement = 'Da sinistra a destra'
                elif hand_pos[0] - prev_hand_pos[0] < -movement_threshold:
                    hand_movement = 'Da destra a sinistra'
                if hand_pos[1] - prev_hand_pos[1] > movement_threshold:
                    hand_movement = 'In basso'
                elif hand_pos[1] - prev_hand_pos[1] < -movement_threshold:
                    hand_movement = 'In alto'

            # Aggiorna la variabile prev_hand_pos
            prev_hand_pos = hand_pos

            # Calcola la distanza tra il polso e la punta dell’indice
            wrist = hand_landmarks.landmark[0]
            index_tip = hand_landmarks.landmark[8]
            distance = np.sqrt((wrist.x - index_tip.x) ** 2 + (wrist.y - index_tip.y) ** 2)

            # Normalizza il valore della distanza in un intervallo da 0 a 1
            distance = min(max(distance, 0), 1)

        window['-HAND_DETECTION_STATUS-'].update(value="")
    else:
        window['-HAND_DETECTION_STATUS-'].update(value="Mano non rilevata")

    # Aggiorna il testo nella colonna di sinistra
    window['-MOVEMENT-'].update(value=f"Movimento: {hand_movement}")

    # Aggiorna la lista last_movements con il nuovo movimento rilevato
    if hand_movement != 'Nessun movimento':
        last_movements.append(hand_movement)
        if len(last_movements) > 5:
            last_movements.pop(0)

    # Aggiorna il testo nella colonna di sinistra con gli ultimi 5 movimenti
    movements_text = "\n".join([f"{i + 1}. {movement}" for i, movement in enumerate(last_movements)])
    window['-LAST_MOVEMENTS-'].update(value=f"Ultimi 5 movimenti:\n{movements_text}")

    # Aggiorna la barra di avanzamento della luminosità ambientale
    window['-BRIGHTNESS_BAR-'].update(int(brightness / 255 * 100))

    # Aggiorna la barra di avanzamento della distanza ottimale con il valore normalizzato della distanza
    window['-DISTANCE_BAR-'].update(int(distance * 100))

    # Aggiorna l'immagine nella colonna di destra
    window['-RIGHT_IMAGE-'].update(data=cv2.imencode('.png', image)[1].tobytes())

    # Gestisci gli eventi della finestra PySimpleGUI
    event, values = window.read(timeout=20)
    if event == 'Esci' or event == sg.WIN_CLOSED:
        break

cap.release()
window.close()

