import cv2
import mediapipe as mp
import time

camera = cv2.VideoCapture(0)
mpMao  = mp.solutions.hands
maos   = mpMao.Hands()
mpDesenhoMao = mp.solutions.drawing_utils

tic = 0
tac = 0

while True:
    sucesso, imagem = camera.read()
    imagemRGB  = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
    resultMaos = maos.process(imagemRGB)

    if resultMaos.multi_hand_landmarks:
        for maosPntRef in resultMaos.multi_hand_landmarks:
            mpDesenhoMao.draw_landmarks(imagem, maosPntRef, mpMao.HAND_CONNECTIONS)

    tac = time.time()
    fps = 1/(tac-tic)
    tic = tac

    cv2.putText(imagem, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)

    cv2.imshow("Camera", imagem)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break