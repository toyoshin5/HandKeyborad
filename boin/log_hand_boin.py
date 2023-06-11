# Description: 手のランドマークをcsvファイルに書き込んで機械学習のデータセットを作成するプログラム

import mediapipe as mp
import cv2
import serial
import numpy as np

target_dict = {0:"あ", 1:"い", 2:"う", 3:"え", 4:"お"}
rev_target_dict = {"あ":0, "い":1, "う":2, "え":3, "お":4}

DATANUM_OF_GYO = 200 #1段あたりのデータ数
ARDUINO_PATH = "/dev/tty.usbmodem1301" #Arduinoのシリアルポート
VIDEOCAPTURE_NUM = 0 #ビデオキャプチャの番号

def write_header(f):
    s = "target,"
    for i in range(5):
        s += ("x" + str(i) + ",y" + str(i) + ",")
    s = s[:-1]
    f.write(s + "\n")
    return
#各landmarkのx,y座標をカンマ区切りでまとめる
def write_csv(f,pos,target):
    s = target + ","
    for i in range(5):
        s += str(pos[i][0]) + "," + str(pos[i][1]) + ","
    s = s[:-1]
    f.write(s + "\n")
    return


#main

if __name__ == "__main__":
    #シリアル
    ser = serial.Serial(ARDUINO_PATH, 9600)
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
    cap = cv2.VideoCapture(VIDEOCAPTURE_NUM)
    #csvファイルに書き込み
    f = open('hand_landmark_boin.csv', 'a')  
    #ファイルが空の場合はヘッダーを書き込み
    if f.tell() == 0:
        write_header(f)
    cnt = 0
    #押下時の親指のx,y座標を格納
    thumb_tap = np.array([0,0])
    thumb_tap_org = np.array([0,0])
    #リリース時から直近5フレーム親指のx,y座標を格納(前が最新)
    thumb_release = np.array([[0,0],[0,0],[0,0],[0,0],[0,0]])
    #押下の状態
    is_tap = False
    while cap.isOpened():
        if cnt == DATANUM_OF_GYO:
            break
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            origin = [(hand_landmarks.landmark[8].x+hand_landmarks.landmark[12].x+hand_landmarks.landmark[16].x+hand_landmarks.landmark[10].x)/4,(hand_landmarks.landmark[8].y+hand_landmarks.landmark[12].y+hand_landmarks.landmark[16].y+hand_landmarks.landmark[10].y)/4]
            if is_tap:
                #親指のx,y座標を格納
                thumb  = [hand_landmarks.landmark[4].x,hand_landmarks.landmark[4].y]
                thumb = [thumb[0]-origin[0],thumb[1]-origin[1]]
                thumb_release = np.append(thumb_release,[thumb],axis=0)
                if (len(thumb_release) > 5) :
                    thumb_release = np.delete(thumb_release,0,0)

            if ser.in_waiting > 0:
                thumb  = [hand_landmarks.landmark[4].x,hand_landmarks.landmark[4].y]
                thumb = [thumb[0]-origin[0],thumb[1]-origin[1]]
                row = ser.readline()
                msg = row.decode('utf-8').rstrip()
                if msg == "tap":
                    #親指の押下時のx,y座標を格納
                    thumb_tap_org = [hand_landmarks.landmark[4].x,hand_landmarks.landmark[4].y]
                    thumb_tap = np.array(thumb_tap_org)-np.array(origin)
                    #thum_releaseの各要素をthumb_tapで初期化
                    thumb_release = np.array([thumb_tap,thumb_tap,thumb_tap,thumb_tap,thumb_tap])
                    is_tap = True
                if msg == "release":
                    #リリース時から直近5フレームのx,y座標から親指の押下時のx,y座標を引いて移動量を計算
                    thumb_move = np.array(thumb_release)-np.array(thumb_tap)
                    #中指のむきに従って、回転
                    a = np.array([hand_landmarks.landmark[9].x-hand_landmarks.landmark[12].x,hand_landmarks.landmark[9].y-hand_landmarks.landmark[12].y])
                    sin_land = a[1]/np.sqrt(a[0]**2+a[1]**2)
                    cos_land = a[0]/np.sqrt(a[0]**2+a[1]**2)
                    #回転行列を計算
                    rot_mat = np.array([[cos_land,sin_land],[-sin_land,cos_land]])
                    #親指の移動量を回転
                    thumb_move = np.dot(rot_mat,thumb_move.T).T
                    print(thumb_move)
                    #csvファイルに書き込み
                    write_csv(f,thumb_move,target)

                    cnt += 1
                    is_tap = False
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    hands.close()
    cap.release()
    f.close()
