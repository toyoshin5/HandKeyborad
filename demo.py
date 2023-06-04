# -*- coding: utf-8 -*-
# Description: 手のランドマークをXGBoostのモデルに入力して、子音を予測するプログラム

from time import sleep
import mediapipe as mp
import numpy as np
import pandas as pd
import cv2
from xgboost import XGBClassifier
import pickle 
import serial
import sys

#image/hiragana_img_manager.pyをimport
sys.path.append("image")
from hiragana_img_manager import HiraganaImgManager
#描画用インスタンス
im = HiraganaImgManager()

MODE = "2D" #2D or 3D
ARDUINO_PATH = "/dev/tty.usbmodem11101" #Arduinoのシリアルポート
VIDEOCAPTURE_NUM = 0 #ビデオキャプチャの番号

target_dict = {0:"あ",1:"か",2:"さ",3:"た",4:"な",5:"は",6:"ま",7:"や",8:"ら",9:"わ",10:"だ"}
rev_target_dict = {"あ":0,"か":1,"さ":2,"た":3,"な":4,"は":5,"ま":6,"や":7,"ら":8,"わ":9,"だ":10}

def showHiragana(char, shiin ,img,size,hand_landmark,res):
    #辞書を作成
    shiin_center_dict = {}
    shiin_center_dict["あ"] = [((hand_landmark[8].x+hand_landmark[7].x)/2),((hand_landmark[8].y+hand_landmark[7].y)/2)]
    shiin_center_dict["か"] = [((hand_landmark[7].x+hand_landmark[6].x)/2),((hand_landmark[7].y+hand_landmark[6].y)/2)]
    shiin_center_dict["さ"] = [((hand_landmark[6].x+hand_landmark[5].x)/2),((hand_landmark[6].y+hand_landmark[5].y)/2)]
    shiin_center_dict["た"] = [((hand_landmark[12].x+hand_landmark[11].x)/2),((hand_landmark[12].y+hand_landmark[11].y)/2)]
    shiin_center_dict["な"] = [((hand_landmark[11].x+hand_landmark[10].x)/2),((hand_landmark[11].y+hand_landmark[10].y)/2)]
    shiin_center_dict["は"] = [((hand_landmark[10].x+hand_landmark[9].x)/2),((hand_landmark[10].y+hand_landmark[9].y)/2)]
    shiin_center_dict["ま"] = [((hand_landmark[16].x+hand_landmark[15].x)/2),((hand_landmark[16].y+hand_landmark[15].y)/2)]
    shiin_center_dict["や"] = [((hand_landmark[15].x+hand_landmark[14].x)/2),((hand_landmark[15].y+hand_landmark[14].y)/2)]
    shiin_center_dict["ら"] = [((hand_landmark[14].x+hand_landmark[13].x)/2),((hand_landmark[14].y+hand_landmark[13].y)/2)]
    shiin_center_dict["だ"] = [((hand_landmark[20].x+hand_landmark[19].x)/2),((hand_landmark[20].y+hand_landmark[19].y)/2)]
    shiin_center_dict["わ"] = [((hand_landmark[19].x+hand_landmark[18].x)/2),((hand_landmark[19].y+hand_landmark[18].y)/2)]
    #解像度に合わせて座標を変換
    for key in shiin_center_dict:
        shiin_center_dict[key] = (int(shiin_center_dict[key][0]*res[0]),int(shiin_center_dict[key][1]*res[1]))
    pos = [shiin_center_dict[shiin][0],shiin_center_dict[shiin][1]]
    #画像合成
    img = im.putHiragana(char,img,pos,size)
    return img

   

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
    #解像度
    resolution = [cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)]
    #シリアル通信の設定
    ser = serial.Serial(ARDUINO_PATH, 9600)
    #入力された母音
    shiin = ""
    #50音データの読み込み
    gojuon_data = pd.read_csv("50on.csv",encoding="UTF-8", header=None)
    #押下時の親指のx,y座標を格納
    thumb_tap = np.array([0,0])
    thumb_tap_org = np.array([0,0])
    #リリース時の親指のx,y座標を格納
    thumb_release = np.array([0,0])
    thumb_release_org = np.array([0,0])
    #入力表示用のカウンター
    count = 0
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("カメラから映像を取得できませんでした")
            continue
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
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
            if ser.in_waiting > 0:
                row = ser.readline()
                msg = row.decode('utf-8').rstrip()
                if msg == "tap":
                    #母音を判定
                    pred = shiin_predict(model,hand_landmarks.landmark,MODE)
                    shiin = target_dict[pred[0]]
                    #親指の押下時のx,y座標を格納
                    thumb_tap_org = [hand_landmarks.landmark[4].x,hand_landmarks.landmark[4].y]
                    thumb_tap = np.array(thumb_tap_org)-np.array(origin)
                    print(shiin,end=" ")
                if msg == "release":
                    #親指のリリース時のx,y座標を格納
                    thumb_release_org = [hand_landmarks.landmark[4].x,hand_landmarks.landmark[4].y]
                    thumb_release = np.array(thumb_release_org)-np.array(origin)
                    #親指の動きを計算
                    thumb_move = np.array(thumb_release)-np.array(thumb_tap)
                    boin_num = 0
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
                        if theta < 45 or theta > 315:
                            print("左",end=" ")
                            boin_num = 1
                        elif theta < 135:
                            print("上",end=" ")
                            boin_num = 2
                        elif theta < 225:
                            print("右",end=" ")
                            boin_num = 3
                        else:
                            print("下",end=" ")
                            boin_num = 4

                    #ひらがなを判定。
                    #1列目がshiinと一致する行
                    gojuon_data_boin = gojuon_data[gojuon_data[0] == shiin]
                    #その行のshiin_num列
                    hiragana = gojuon_data_boin[boin_num].values[0]
                    if hiragana != "*":
                        #入力文字列に追加
                        print(hiragana)
                    count = 5
            if count > 0:
                image = showHiragana(hiragana, shiin ,image, 200,hand_landmarks.landmark,resolution)
        #FPSを表示   
        fps = cap.get(cv2.CAP_PROP_FPS)
        cv2.putText(image, str(fps) + "fps", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)
        cv2.imshow('MediaPipe Hands', image)
        key = cv2.waitKey(1)  
        count -= 1;             
    cap.release()
