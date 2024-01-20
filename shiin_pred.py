# Description: 手のランドマークをXGBoostのモデルに入力して、子音を予測するプログラム

import sys
import mediapipe as mp
import numpy as np
import pandas as pd
import cv2
import pickle 

#image/hiragana_img_manager.pyをimport
sys.path.append("image")
from hiragana_img_manager import HiraganaImgManager

MODE = "3D" #2D or 3D
VIDEOCAPTURE_NUM = 0 #ビデオキャプチャの番号
USE_ML = False
target_dict = {0:"あ",1:"か",2:"さ",3:"た",4:"な",5:"は",6:"ま",7:"や",8:"ら",9:"わ",10:"小"}
#2D
def rotate_coordinates(rotation_matrix, coordinates):
    rotated_coordinates = np.einsum('ijk,jk->ik', rotation_matrix, coordinates)
    return np.array(rotated_coordinates[0]), np.array(rotated_coordinates[1])
#2D
def shiin_predict_noML(landmark,mode):
    midpoint = []
    for i in range(5,16,4):
        #中点(?)を計算
        if mode == "2D":
            midpoint.append([(landmark[i+2].x+landmark[i+3].x)/2,(landmark[i+2].y+landmark[i+3].y)/2])
            midpoint.append([(landmark[i+1].x+landmark[i+2].x)/2,(landmark[i+1].y+landmark[i+2].y)/2])
            midpoint.append([(landmark[i].x+landmark[i+1].x*2)/3,(landmark[i].y+landmark[i+1].y*2)/3])
        elif mode == "3D":
            midpoint.append([(landmark[i+2].x+landmark[i+3].x)/2,(landmark[i+2].y+landmark[i+3].y)/2,(landmark[i+2].z+landmark[i+3].z)/2])
            midpoint.append([(landmark[i+1].x+landmark[i+2].x)/2,(landmark[i+1].y+landmark[i+2].y)/2,(landmark[i+1].z+landmark[i+2].z)/2])
            midpoint.append([(landmark[i].x+landmark[i+1].x*2)/3,(landmark[i].y+landmark[i+1].y*2)/3,(landmark[i].z+landmark[i+1].z*2)/3])
    if mode == "2D":
        midpoint.append([(landmark[18].x+landmark[19].x)/2,(landmark[18].y+landmark[19].y)/2]) #わ
        midpoint.append([(landmark[19].x+landmark[20].x)/2,(landmark[19].y+landmark[20].y)/2]) #小
    elif mode == "3D":
        midpoint.append([(landmark[18].x+landmark[19].x)/2,(landmark[18].y+landmark[19].y)/2,(landmark[18].z+landmark[19].z)/2])
        midpoint.append([(landmark[19].x+landmark[20].x)/2,(landmark[19].y+landmark[20].y)/2,(landmark[19].z+landmark[20].z)/2])
    #親指の先端の座標
    thumb = [landmark[4].x,landmark[4].y,landmark[4].z]
    #最も近い中点を探す
    min_dist = 100000
    min_index = 0
    for i in range(len(midpoint)):
        if mode == "3D":
            dist = np.sqrt((thumb[0]-midpoint[i][0])**2+(thumb[1]-midpoint[i][1])**2+(thumb[2]-midpoint[i][2])**2)
        elif mode == "2D":
            dist = np.sqrt((thumb[0]-midpoint[i][0])**2+(thumb[1]-midpoint[i][1])**2)

        if dist < min_dist:
            min_dist = dist
            min_index = i
    return min_index
#ヨー方向に回転する関数。3Dで使用
def rotate_points_yaw_pitch(coords):
    #  指の平面を求める
    #5~20の点
    finger_coords = coords[5:21]
    # 最小二乗法による平面の方程式の解を求める
    A = np.column_stack((finger_coords[:, 0], finger_coords[:, 1], np.ones(len(finger_coords))))
    b = finger_coords[:, 2]
    coefficients, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    # 平面の方程式の係数を取得
    a, b ,_= coefficients
    # 平面の法線ベクトルを計算
    normal_vector = np.array([a, b, -1])
    # 法線ベクトルを正規化
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    #yaw方向
    yaw_theta = np.arctan2(normal_vector[0], normal_vector[2])
    yaw_theta = -yaw_theta
    # 回転行列を計算
    rotation_matrix_yaw = np.array([[np.cos(yaw_theta),0, np.sin(yaw_theta)],
                                    [0, 1, 0],
                                    [-np.sin(yaw_theta), 0, np.cos(yaw_theta)]])
    # 座標の回転
    rotated_coords = np.dot(rotation_matrix_yaw, coords.T).T
    # #pitch方向
    pitch_theta = np.arctan2(normal_vector[1], normal_vector[2])
    pitch_theta = -pitch_theta
    # 回転行列を計算
    rotation_matrix_pitch = np.array([[1, 0, 0],
                                        [0, np.cos(pitch_theta), -np.sin(pitch_theta)],
                                        [0, np.sin(pitch_theta), np.cos(pitch_theta)]])
    # 座標の回転
    rotated_coords = np.dot(rotation_matrix_pitch, rotated_coords.T).T
    return rotated_coords

#ロール方向に回転する関数。3Dで使用。各指が左を向くようにする
def rotate_points_roll(coords):
    #8->5, 12->9, 16->13, 20->17のベクトルを計算
    vector1 = -np.array([coords[8][0]-coords[5][0], coords[8][1]-coords[5][1]])
    vector2 = -np.array([coords[12][0]-coords[9][0], coords[12][1]-coords[9][1]])
    vector3 = -np.array([coords[16][0]-coords[13][0], coords[16][1]-coords[13][1]])
    vector4 = -np.array([coords[20][0]-coords[17][0], coords[20][1]-coords[17][1]])
    #平均をとる
    vector = (vector1 + vector2 + vector3 + vector4) / 4
    #roll方向
    roll_theta = np.arctan2(vector[1], vector[0])
    roll_theta = -roll_theta
    # 回転行列を計算
    rotation_matrix_roll = np.array([[np.cos(roll_theta), -np.sin(roll_theta), 0],
                                    [np.sin(roll_theta), np.cos(roll_theta), 0],
                                    [0, 0, 1]])
    # 座標の回転
    rotated_coords = np.dot(rotation_matrix_roll, coords.T).T
    return rotated_coords

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
            hand_size = np.sqrt((landmark_dict['x0']-landmark_dict['x17'])**2+(landmark_dict['y0']-landmark_dict['y17'])**2)
            # 4から各点までの変位を特徴量に追加
            landmark_dict = pd.concat([landmark_dict, pd.DataFrame({
                'offset_x'+str(i): (landmark_dict['x4'] - landmark_dict['x'+str(i)]) / hand_size,
                'offset_y'+str(i): (landmark_dict['y4'] - landmark_dict['y'+str(i)]) / hand_size,
                'distance'+str(i): np.sqrt((landmark_dict['x4'] - landmark_dict['x'+str(i)])**2+(landmark_dict['y4'] - landmark_dict['y'+str(i)])**2)/ hand_size
            })], axis=1)
    elif mode == "3D":
            #回転変換を行う
            #各成分の配列に変換
            x_coords = []
            y_coords = []
            z_coords = []
            for i in range(21):
                x_coords.append(landmark_dict['x'+str(i)])
                y_coords.append(landmark_dict['y'+str(i)])
                z_coords.append(landmark_dict['z'+str(i)])
            coords = np.array([x_coords,y_coords,z_coords]).T
            #回転
            rotated_coords = rotate_points_yaw_pitch(coords)
            rotated_coords = rotate_points_roll(rotated_coords)
            #モデルで使得る形に戻す
            for i in range(21):
                landmark_dict['x'+str(i)],landmark_dict['y'+str(i)],landmark_dict['z'+str(i)] = rotated_coords[i][0],rotated_coords[i][1],rotated_coords[i][2]
            # 列を結合する
            landmark_dict = pd.concat([landmark_dict, pd.DataFrame({
                'offset_x'+str(i): (landmark_dict['x4'] - landmark_dict['x'+str(i)])/ hand_size,
                'offset_y'+str(i): (landmark_dict['y4'] - landmark_dict['y'+str(i)])/ hand_size,
                'offset_z'+str(i): (landmark_dict['z4'] - landmark_dict['z'+str(i)])/ hand_size,
                'distance'+str(i): np.sqrt((landmark_dict['x4'] - landmark_dict['x'+str(i)])**2+(landmark_dict['y4'] - landmark_dict['y'+str(i)])**2+(landmark_dict['z4'] - landmark_dict['z'+str(i)])**2)/ hand_size
            })], axis=1) #xn,ynを消去
    for i in range(0,33):
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
        min_tracking_confidence=0.3,
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
            if USE_ML:
                pred = shiin_predict(model,hand_landmarks.landmark,MODE)[0]
            else:
                pred = shiin_predict_noML(hand_landmarks.landmark,MODE)
            #予測結果を画面に日本語で大きく表示
            #真ん中に画像(../image/test.png)を表示
            image = im.putHiragana(target_dict[pred],image,[50,50],400)
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27: #ESCキーで終了
            break

    hands.close()
    cap.release()
