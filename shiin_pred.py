# Description: 手のランドマークをXGBoostのモデルに入力して、子音を予測するプログラム

import sys
import mediapipe as mp
import numpy as np
import pandas as pd
import cv2
from xgboost import XGBClassifier
import pickle 

#image/hiragana_img_manager.pyをimport
sys.path.append("image")
from hiragana_img_manager import HiraganaImgManager

MODE = "2D" #2D or 3D
VIDEOCAPTURE_NUM = 0 #ビデオキャプチャの番号

target_dict = {0:"あ",1:"か",2:"さ",3:"た",4:"な",5:"は",6:"ま",7:"や",8:"ら",9:"わ",10:"小"}

def rotate_coordinates(rotation_matrix, coordinates):
    rotated_coordinates = np.einsum('ijk,jk->ik', rotation_matrix, coordinates)
    return np.array(rotated_coordinates[0]), np.array(rotated_coordinates[1])


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

    #4から各点までの変位を特徴量に追加
    hand_size = np.sqrt((landmark_dict['x0']-landmark_dict['x17'])**2+(landmark_dict['y0']-landmark_dict['y17'])**2)
    for i in range(1,21):
        if mode == "2D":
            landmark_dict['offset_x'+str(i)] = (landmark_dict['x4']-landmark_dict['x'+str(i)])/hand_size
            landmark_dict['offset_y'+str(i)] = (landmark_dict['y4']-landmark_dict['y'+str(i)])/hand_size
        elif mode == "3D":
            landmark_dict['offset_x'+str(i)] = (landmark_dict['x4']-landmark_dict['x'+str(i)])/hand_size
            landmark_dict['offset_y'+str(i)] = (landmark_dict['y4']-landmark_dict['y'+str(i)])/hand_size
            landmark_dict['offset_z'+str(i)] = (landmark_dict['z4']-landmark_dict['z'+str(i)])/hand_size
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
    #ImageManagerのインスタンス生成
    im = HiraganaImgManager()
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
        image = cv2.flip(image, 1)
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
            #予測結果を画面に日本語で大きく表示
            #真ん中に画像(../image/test.png)を表示
            image = im.putHiragana(target_dict[pred[0]],image,[50,50],400)
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27: #ESCキーで終了
            break

    hands.close()
    cap.release()
