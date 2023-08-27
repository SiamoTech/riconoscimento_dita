# Importa il modulo OpenCV (cv2) per l'elaborazione delle immagini e dei video
import cv2
# Importa il modulo MediaPipe (mp) per il rilevamento delle mani
import mediapipe as mp
# Importa il modulo NumPy (np) per il calcolo scientifico
import numpy as np

# Crea un alias per il modulo di disegno di MediaPipe
mp_drawing = mp.solutions.drawing_utils
# Crea un alias per il modulo delle mani di MediaPipe
mp_hands = mp.solutions.hands

# Inizializza la soluzione Hands di MediaPipe
hands = mp_hands.Hands()

# Cattura il video dalla webcam (indice 0)
cap = cv2.VideoCapture(0)

# Crea un dizionario per convertire gli indici delle dita in nomi di dita
finger_dict = {0: 'Pollice', 1: 'Indice', 2: 'Medio', 3: 'Anulare', 4: 'Mignolo'}

# Inizia un ciclo while che continua finché la cattura del video è aperta
while cap.isOpened():
    # Leggi un frame dal video e assegna il risultato a success e image
    success, image = cap.read()
    # Se la lettura del frame non ha avuto successo, interrompi il ciclo while
    if not success:
        break

    # Converti l'immagine in scala di grigi utilizzando la funzione cvtColor di OpenCV
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Calcola la media dei pixel dell'immagine in scala di grigi utilizzando la funzione mean di NumPy
    mean = np.mean(gray)

    # Regola la sensibilità del programma in base alle condizioni di illuminazione dell'immagine catturata dalla webcam
    if mean < 50:
        # Condizioni di scarsa illuminazione
        # Aumenta la sensibilità del programma
        pass
    elif mean > 200:
        # Condizioni di forte illuminazione
        # Riduci la sensibilità del programma
        pass
    else:
        # Condizioni di illuminazione normale
        # Imposta la sensibilità del programma al valore predefinito
        pass

    # Converti l'immagine in RGB utilizzando la funzione cvtColor di OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Rileva le mani nell'immagine utilizzando la soluzione Hands di MediaPipe
    results = hands.process(image)

    # Se sono state rilevate delle mani nell'immagine...
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Disegna i risultati sulle mani rilevate utilizzando il modulo di disegno di MediaPipe
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Inizializza il contatore delle dita alzate a 0
            count = 0
            # Crea una lista vuota per tenere traccia delle dita alzate
            raised_fingers = []

            # Controlla se il pollice è alzato confrontando le coordinate x dei landmark 2, 3 e 4
            if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x < hand_landmarks.landmark[2].x:
                # Se il pollice è alzato, incrementa il contatore delle dita alzate e aggiungi 'Pollice' alla lista delle dita alzate
                count += 1
                raised_fingers.append(finger_dict[0])

            # Controlla se l'indice è alzato confrontando le coordinate y dei landmark 6, 7 e 8
            if hand_landmarks.landmark[8].y < hand_landmarks.landmark[7].y < hand_landmarks.landmark[6].y:
                # Se l'indice è alzato, incrementa il contatore delle dita alzate e aggiungi 'Indice' alla lista delle dita alzate
                count += 1
                raised_fingers.append(finger_dict[1])

            # Controlla se il medio è alzato confrontando le coordinate y dei landmark 10, 11 e 12
            if hand_landmarks.landmark[12].y < hand_landmarks.landmark[11].y < hand_landmarks.landmark[10].y:
                # Se il medio è alzato, incrementa il contatore delle dita alzate e aggiungi 'Medio' alla lista delle dita alzate
                count += 1
                raised_fingers.append(finger_dict[2])

            # Controlla se l'anulare è alzato confrontando le coordinate y dei landmark 14, 15 e 16
            if hand_landmarks.landmark[16].y < hand_landmarks.landmark[15].y < hand_landmarks.landmark[14].y:
                # Se l'anulare è alzato, incrementa il contatore delle dita alzate e aggiungi 'Anulare' alla lista delle dita alzate
                count += 1
                raised_fingers.append(finger_dict[3])

            # Controlla se il mignolo è alzato confrontando le coordinate y dei landmark 18, 19 e 20
            if hand_landmarks.landmark[20].y < hand_landmarks.landmark[19].y < hand_landmarks.landmark[18].y:
                # Se il mignolo è alzato, incrementa il contatore delle dita alzate e aggiungi 'Mignolo' alla lista delle dita alzate
                count += 1
                raised_fingers.append(finger_dict[4])

            # Stampa il numero di dita alzate e quali dita sono alzate sulla console
            print("Numero di dita alzate:", count)
            print("Dita alzate:", ', '.join(raised_fingers))

            # Visualizza il numero di dita alzate e quali dita sono alzate sull'immagine utilizzando la funzione putText di OpenCV
            cv2.putText(image, f"Dita alzate: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(image, f"Dita: {', '.join(raised_fingers)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Visualizza l'immagine con i risultati utilizzando la funzione imshow di OpenCV
    cv2.imshow('Image', image)
    # Attendi 5 millisecondi per un input dall'utente e interrompi il ciclo while se viene premuto il tasto Esc
    if cv2.waitKey(5) & 0xFF == 27:
        break

# Rilascia la cattura del video
cap.release()
