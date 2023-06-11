# -*- coding: utf-8 -*-
# Description: 手のランドマークをXGBoostのモデルに入力して、子音を予測するプログラム

import sys
from time import sleep
import mediapipe as mp
import numpy as np
import pandas as pd
import cv2
from xgboost import XGBClassifier
import pickle 
import serial

MODE = "2D" #2D or 3D
ARDUINO_PATH = "/dev/cu.usbmodem101" #Arduinoのシリアルポート

target_dict = {0:"あ",1:"か",2:"さ",3:"た",4:"な",5:"は",6:"ま",7:"や",8:"ら",9:"わ",10:"小"}
rev_target_dict = {"あ":0,"か":1,"さ":2,"た":3,"な":4,"は":5,"ま":6,"や":7,"ら":8,"わ":9,"小":10}


def shiin_predict(model,landmark,mode):
    #x,yをモデルに入力
    landmark_dict = {}
    for i in range(21):
        landmark_dict['x'+str(i)] = landmark[i].x
        landmark_dict['y'+str(i)] = landmark[i].y
        landmark_dict['z'+str(i)] = landmark[i].z
    landmark_dict = pd.DataFrame(landmark_dict,index=[0]) #1行のデータフレームに変換
    #前処理
    if mode == "2D":
        #回転変換を行う
        a = np.array([landmark_dict['x17']-landmark_dict['x0'],landmark_dict['y17']-landmark_dict['y0']])
        sin_land = a[1]/np.sqrt(a[0]**2+a[1]**2)
        cos_land = a[0]/np.sqrt(a[0]**2+a[1]**2)
        #回転行列を計算
        rot_mat = np.array([[cos_land,sin_land],[-sin_land,cos_land]])

        for i in range(21):
            #x,yを回転変換
            landmark_dict['x'+str(i)],landmark_dict['y'+str(i)] = rotate_coordinates(rot_mat,np.array([landmark_dict['x'+str(i)],landmark_dict['y'+str(i)]]))
        #中点を計算
        cnt = 21
        for i in range(5,20):
            if i%4 != 0:
                landmark_dict["x"+str(cnt)],landmark_dict["y"+str(cnt)],landmark_dict["z"+str(cnt)] = (landmark_dict['x'+str(i)]+landmark_dict['x'+str(i+1)])/2,(landmark_dict['y'+str(i)]+landmark_dict['y'+str(i+1)])/2,(landmark_dict['z'+str(i)]+landmark_dict['z'+str(i+1)])/2
                cnt += 1   
    #4から各点までの変位を特徴量に追加
    hand_size = np.sqrt((landmark_dict['x0']-landmark_dict['x17'])**2+(landmark_dict['y0']-landmark_dict['y17'])**2)
    for i in range(21,32):
        if mode == "2D":
            landmark_dict['offset_x'+str(i)] = (landmark_dict['x4']-landmark_dict['x'+str(i)])/hand_size
            landmark_dict['offset_y'+str(i)] = (landmark_dict['y4']-landmark_dict['y'+str(i)])/hand_size
        elif mode == "3D":
            landmark_dict['offset_x'+str(i)] = (landmark_dict['x4']-landmark_dict['x'+str(i)])/hand_size
            landmark_dict['offset_y'+str(i)] = (landmark_dict['y4']-landmark_dict['y'+str(i)])/hand_size
            landmark_dict['offset_z'+str(i)] = (landmark_dict['z4']-landmark_dict['z'+str(i)])/hand_size
    #xn,ynを消去
    for i in range(0,32):
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
    #入力文字列
    input_str = ""
    #ウェブカメラからの入力
    cap = cv2.VideoCapture(0)
    #シリアル通信の設定
    ser = serial.Serial(ARDUINO_PATH, 9600)
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
        pred = ["*"]
        if results.multi_hand_landmarks:
            #len(results.multi_hand_landmarks) = 写っている手の数
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            pred = shiin_predict(model,hand_landmarks.landmark,MODE)        
        image = putText_japanese(image,input_str,(100,200),100,(255,255,255))
        #FPSを表示   
        fps = cap.get(cv2.CAP_PROP_FPS)
        cv2.putText(image, str(fps) + "fps", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)
        cv2.imshow('MediaPipe Hands', image)
        # キー入力を待機する
        key = cv2.waitKey(1)
        #Arduinoからのシリアル通信を受け取る
        if results.multi_hand_landmarks:
            if ser.in_waiting > 0:
                row = ser.readline()
                msg = row.decode('utf-8').rstrip()
                if msg == "tap":
                    input_str += target_dict[pred[0]]
                    print(input_str)

    cap.release()
