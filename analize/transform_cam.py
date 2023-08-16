

# Description: カメラから骨格を取得し、様々な変換を行って表示するプログラム

import sys
import mediapipe as mp
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
#image/hiragana_img_manager.pyをimport
sys.path.append("image")

VIDEOCAPTURE_NUM = 1 #ビデオキャプチャの番号

#ヨー,ピッチ方向に回転する関数。3Dで使用
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



if __name__ == "__main__":
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
            # mp_drawing.draw_landmarks(
            #     image, hand_landmarks, mp_hands.HAND_CONNECTIONS) 
            x_coords,y_coords,z_coords = [],[],[]
            for i in range(21):
                x_coords.append(hand_landmarks.landmark[i].x)
                y_coords.append(hand_landmarks.landmark[i].y)
                z_coords.append(hand_landmarks.landmark[i].z)
            rotated_landmarks = rotate_points_yaw_pitch(np.array([x_coords,y_coords,z_coords]).T)
            rotated_landmarks = rotate_points_roll(rotated_landmarks)
            #matplotlibで出力
            fig = plt.figure()
            #関節
            ax = fig.add_subplot(111)
            #骨
            bones = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12), (17, 13),(9,13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]
            for bone in bones:
                ax.plot([rotated_landmarks[bone[0]][0], rotated_landmarks[bone[1]][0]], [rotated_landmarks[bone[0]][1], rotated_landmarks[bone[1]][1]], color='blue')
            #y軸反転
            ax.invert_yaxis()
            ax.scatter(rotated_landmarks[:,0], rotated_landmarks[:,1])
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            #plt.show()
            plt.pause(0.05)
            plt.close()


               
            

    hands.close()
    cap.release()

