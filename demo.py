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
ARDUINO_PATH = "/dev/cu.usbmodem213101" #Arduinoのシリアルポート
VIDEOCAPTURE_NUM = 1 #ビデオキャプチャの番号

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
    model = pickle.load(open('shiin/shiin_model_'+MODE+'.pkl', 'rb'))
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
    cap = cv2.VideoCapture(VIDEOCAPTURE_NUM)
    #シリアル通信の設定
    ser = serial.Serial(ARDUINO_PATH, 9600)
    #入力された母音
    boin = ""
    #50音データの読み込み
    gojuon_data = pd.read_csv("50on.csv",encoding="UTF-8", header=None)
    #押下時の親指のx,y座標を格納
    thumb_tap = np.array([0,0])
    thumb_tap_org = np.array([0,0])
    #リリース時の親指のx,y座標を格納
    thumb_release = np.array([0,0])
    thumb_release_org = np.array([0,0])
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
            vec = [hand_landmarks.landmark[12].x-hand_landmarks.landmark[9].x,hand_landmarks.landmark[12].y-hand_landmarks.landmark[9].y]
            origin = [(hand_landmarks.landmark[8].x+hand_landmarks.landmark[12].x+hand_landmarks.landmark[16].x+hand_landmarks.landmark[10].x)/4,(hand_landmarks.landmark[8].y+hand_landmarks.landmark[12].y+hand_landmarks.landmark[16].y+hand_landmarks.landmark[10].y)/4]
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
                    #母音を判定
                    pred = shiin_predict(model,hand_landmarks.landmark,MODE)
                    boin = target_dict[pred[0]]
                    #親指の押下時のx,y座標を格納
                    thumb_tap_org = [hand_landmarks.landmark[4].x,hand_landmarks.landmark[4].y]
                    thumb_tap = np.array(thumb_tap_org)-np.array(origin)
                    print(boin,end=" ")
                if msg == "release":
                    #親指のリリース時のx,y座標を格納
                    thumb_release_org = [hand_landmarks.landmark[4].x,hand_landmarks.landmark[4].y]
                    thumb_release = np.array(thumb_release_org)-np.array(origin)
                    #親指の動きを計算
                    thumb_move = np.array(thumb_release)-np.array(thumb_tap)
                    shiin_num = 0
                    #親指の動きが小さい場合
                    if np.linalg.norm(thumb_move) < 0.02:
                        print("中",end=" ")
                    else:
                        #上下左右を判定
                        cos_theta = np.dot(vec,thumb_move)/(np.linalg.norm(vec)*np.linalg.norm(thumb_move))
                        gaiseki = np.cross(vec,thumb_move,)
                        if(gaiseki > 0):
                            theta = np.arccos(cos_theta)*180/np.pi
                        else:
                            theta = 360-np.arccos(cos_theta)*180/np.pi
                        if theta < 45 or theta > 305:
                            print("左",end=" ")
                            shiin_num = 1
                        elif theta < 135:
                            print("下",end=" ")
                            shiin_num = 4
                        elif theta < 225:
                            print("右",end=" ")
                            shiin_num = 3
                        else:
                            print("上",end=" ")
                            shiin_num = 2
                    #ひらがなを判定。
                    #1列目がboinと一致する行
                    gojuon_data_boin = gojuon_data[gojuon_data[0] == boin]
                    #その行のshiin_num列
                    hiragana = gojuon_data_boin[shiin_num].values[0]
                    if hiragana != "*":
                        #入力文字列に追加
                        print(hiragana)


                        
    cap.release()
