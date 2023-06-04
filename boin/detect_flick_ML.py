import mediapipe as mp
import numpy as np
import cv2
import serial
import pandas as pd

ARDUINO_PATH = "/dev/cu.usbmodem11101" #Arduinoのシリアルポート

target_dict = {0:"あ", 1:"い", 2:"う", 3:"え", 4:"お"}
rev_target_dict = {"あ":0, "い":1, "う":2, "え":3, "お":4}

def shiin_predict(model,pos):
    #x,yをモデルに入力
    pos_dict = {}
    for i in range(5):
        pos_dict['x'+str(i)] = pos[i][0]
        pos_dict['y'+str(i)] = pos[i][1]
    pos_dict = pd.DataFrame(pos_dict,index=[0]) #1行のデータフレームに変換
    #前処理
    #予測
    pred = model.predict(pos_dict)
    return pred

if __name__ == '__main__':
    #モデル読み込み
    model = pd.read_pickle("boin_model_2D.pkl")
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    mp_drawing = mp.solutions.drawing_utils
    #ウェブカメラからの入力
    cap = cv2.VideoCapture(0)
    resolution = [cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)]
    #シリアル通信の設定
    ser = serial.Serial(ARDUINO_PATH, 9600)
    #押下時の親指のx,y座標を格納
    thumb_tap = np.array([0,0])
    thumb_tap_org = np.array([0,0])
    #リリース時から直近5フレーム親指のx,y座標を格納(前が最新)
    thumb_release = np.array([[0,0],[0,0],[0,0],[0,0],[0,0]])
    #押下の状態
    is_tap = False
    while cap.isOpened():
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
                    pred = shiin_predict(model,thumb_move)
                    print(target_dict[pred[0]])
                    is_tap = False
        cv2.imshow('MediaPipe Hands', image)
        kry = cv2.waitKey(1)

    hands.close()
    cap.release()
