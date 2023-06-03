# Description: 手のランドマークをcsvファイルに書き込んで機械学習のデータセットを作成するプログラム

import mediapipe as mp
import cv2

target_dict = {0:"あ", 1:"い", 2:"う", 3:"え", 4:"お"}
rev_target_dict = {"あ":0, "い":1, "う":2, "え":3, "お":4}

DATANUM_OF_GYO = 9000 #1段あたりのデータ数

def write_header(f):
    s = "target,"
    for i in range(21):
        s += ("x" + str(i) + ",y" + str(i) + ",")
    s = s[:-1]
    f.write(s + "\n")
    return
#各landmarkのx,y座標をカンマ区切りでまとめる
def write_csv(f,landmark,target):
    s = target + ","
    for i in range(21):
        s += str(landmark[i].x) + "," + str(landmark[i].y) + ","
    s = s[:-1]
    f.write(s + "\n")
    return


#main
if __name__ == "__main__":
    #ターゲットの段の入力
    dan = input("(あ~お)の行を入力:")
    target = str(rev_target_dict[dan])
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    mp_drawing = mp.solutions.drawing_utils
    #ウェブカメラからの入力
    cap = cv2.VideoCapture(1)
    #csvファイルに書き込み
    f = open('hand_landmark_boin.csv', 'a')  
    #ファイルが空の場合はヘッダーを書き込み
    if f.tell() == 0:
        write_header(f)
    cnt = 0
    while cap.isOpened():
        if cnt == DATANUM_OF_GYO:
            break
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            #csvファイルに書き込み
            write_csv(f,hand_landmarks.landmark,target)
            cnt += 1
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
        print("\r" + str(cnt),end="")

    hands.close()
    cap.release()
    f.close()
