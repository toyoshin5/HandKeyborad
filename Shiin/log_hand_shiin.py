# Description: 手のランドマークをcsvファイルに書き込んで機械学習のデータセットを作成するプログラム

import mediapipe as mp
import cv2

target_dict = {0:"あ",1:"か",2:"さ",3:"た",4:"な",5:"は",6:"ま",7:"や",8:"ら",9:"わ",10:"だ"}
rev_target_dict = {"あ":0,"か":1,"さ":2,"た":3,"な":4,"は":5,"ま":6,"や":7,"ら":8,"わ":9,"だ":10}

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
    print(s)
    f.write(s + "\n")
    return


#main
if __name__ == "__main__":
    #ターゲットの段の入力
    dan = input("(あ~わ)の段を入力:")
    target = str(rev_target_dict[dan])
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    mp_drawing = mp.solutions.drawing_utils
    #ウェブカメラからの入力
    cap = cv2.VideoCapture(0)
    #csvファイルに書き込み
    f = open('hand_landmark.csv', 'a')  
    #ファイルが空の場合はヘッダーを書き込み
    if f.tell() == 0:
        write_header(f)

    cnt = 0
    while cap.isOpened():
        if cnt == 500:
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
            #len(results.multi_hand_landmarks) = 写っている手の数
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            #csvファイルに書き込み
            if cnt % 5 == 0:
                write_csv(f,hand_landmarks.landmark,target)
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
        cnt += 1

    hands.close()
    cap.release()
    f.close()
