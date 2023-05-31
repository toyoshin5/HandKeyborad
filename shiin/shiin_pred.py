# Description: 手のランドマークをXGBoostのモデルに入力して、子音を予測するプログラム

import sys
import mediapipe as mp
import numpy as np
import pandas as pd
import cv2
from xgboost import XGBClassifier
import pickle 


MODE = "2D" #2D or 3D
VIDEOCAPTURE_NUM = 0 #ビデオキャプチャの番号

target_dict = {0:"あ",1:"か",2:"さ",3:"た",4:"な",5:"は",6:"ま",7:"や",8:"ら",9:"わ",10:"だ"}
rev_target_dict = {"あ":0,"か":1,"さ":2,"た":3,"な":4,"は":5,"ま":6,"や":7,"ら":8,"わ":9,"だ":10}

def shiin_predict(model,landmark,mode):
    #x,yをモデルに入力
    landmark_dict = {}
    for i in range(21):
        landmark_dict['x'+str(i)] = landmark[i].x
        landmark_dict['y'+str(i)] = landmark[i].y
        if mode == "3D":
            landmark_dict['z'+str(i)] = landmark[i].z
    landmark_dict = pd.DataFrame(landmark_dict,index=[0]) #1行のデータフレームに変換
    #前処理
    #4から各点までの距離を特徴量に追加
    hand_size = np.sqrt((landmark_dict['x0']-landmark_dict['x17'])**2+(landmark_dict['y0']-landmark_dict['y17'])**2)
    for i in range(5,21):
        if mode == "2D":
            landmark_dict['distance'+str(i)] = np.sqrt((landmark_dict['x4']-landmark_dict['x'+str(i)])**2+(landmark_dict['y4']-landmark_dict['y'+str(i)])**2)/hand_size
        elif mode == "3D":
            landmark_dict['distance'+str(i)] = np.sqrt((landmark_dict['x4']-landmark_dict['x'+str(i)])**2+(landmark_dict['y4']-landmark_dict['y'+str(i)])**2+(landmark_dict['z4']-landmark_dict['z'+str(i)])**2)/hand_size
    #xn,ynを消去
    for i in range(0,21):
        if mode == "2D":
            landmark_dict = landmark_dict.drop(['x'+str(i),'y'+str(i)],axis=1)
        elif mode == "3D":
            landmark_dict = landmark_dict.drop(['x'+str(i),'y'+str(i),'z'+str(i)],axis=1)
    #予測
    pred = model.predict(landmark_dict)
    return pred

#日本語を表示する関数
from PIL import Image, ImageDraw, ImageFont
def putText_japanese(img, text, point, size, color):
    #hiragino font
    try:
        fontpath = '/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc'
        font = ImageFont.truetype(fontpath, size)
    except:
        print("フォントが見つかりませんでした。: " + fontpath)
        sys.exit()
    #imgをndarrayからPILに変換
    img_pil = Image.fromarray(img)
    #drawインスタンス生成
    draw = ImageDraw.Draw(img_pil)
    #テキスト描画
    draw.text(point, text, fill=color, font=font)
    #PILからndarrayに変換して返す
    return np.array(img_pil)

#main
if __name__ == "__main__":
    #モデルの読み込み
    model = pickle.load(open('shiin_model_'+MODE+'.pkl', 'rb'))
    #ターゲットの段の入力
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    mp_drawing = mp.solutions.drawing_utils
    #ウェブカメラからの入力
    cap = cv2.VideoCapture(VIDEOCAPTURE_NUM)
    while cap.isOpened():
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
            #予測
            pred = shiin_predict(model,hand_landmarks.landmark,MODE)
            print(target_dict[pred[0]])
            #予測結果を画面に日本語で大きく表示
            image = putText_japanese(image,target_dict[pred[0]],(100,200),200,(255,255,255))
        #FPSを表示
        fps = cap.get(cv2.CAP_PROP_FPS)
        cv2.putText(image, str(fps) + "fps", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27: #ESCキーで終了
            break

    hands.close()
    cap.release()
