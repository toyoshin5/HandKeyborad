

import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd

def calc_regression_plane(coords):
    # 最小二乗法による平面の方程式の解を求める
    A = np.column_stack((coords[:, 0], coords[:, 1], np.ones(len(coords))))
    b = coords[:, 2]
    coefficients, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    # 平面の方程式の係数を取得
    a, b ,c = coefficients
    return a, b, c

def rotate_points_yaw_pitch(coords):
    #  指の平面を求める
    #5~20の点
    finger_coords = coords[5:21]
    # 最小二乗法による平面の方程式の解を求める
    a,b,_ = calc_regression_plane(finger_coords)
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

#ロール方向に回転する関数。各指が左を向くようにする
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

def showPlotProjected(x,y):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x, y,c = 'r')
    #000111222
    x_coords = [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3]
    y_coords = [0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]
    ax.scatter(x_coords, y_coords, c = 'b')
    
     # スケールの調整xは-1~4, yは-1~4
    ax.set_xlim(-1, 4)
    

    #y軸は下向きに正
    ax.invert_yaxis()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.show()

#4つの点から射影変換行列を求める関数
def find_homography(src, dst):
    
    # X = (x*h0 +y*h1 + h2) / (x*h6 + y*h7 + 1)
    # Y = (x*h3 +y*h4 + h5) / (x*h6 + y*h7 + 1)
    #
    # X = (x*h0 +y*h1 + h2) - x*h6*X - y*h7*X
    # Y = (x*h3 +y*h4 + h5) - x*h6*Y - y*h7*Y
    
    x1, y1 = src[0]
    x2, y2 = src[1]
    x3, y3 = src[2]
    x4, y4 = src[3]
    
    u1, v1 = dst[0]
    u2, v2 = dst[1]
    u3, v3 = dst[2]
    u4, v4 = dst[3]
    
    A = np.matrix([
            [ x1, y1, 1, 0, 0, 0, -x1*u1, -y1*u1, 0 ],
            [ 0, 0, 0, x1, y1, 1, -x1*v1, -y1*v1, 0 ],
            [ x2, y2, 1, 0, 0, 0, -x2*u2, -y2*u2, 0 ],
            [ 0, 0, 0, x2, y2, 1, -x2*v2, -y2*v2, 0 ],
            [ x3, y3, 1, 0, 0, 0, -x3*u3, -y3*u3, 0 ],
            [ 0, 0, 0, x3, y3, 1, -x3*v3, -y3*v3, 0 ],
            [ x4, y4, 1, 0, 0, 0, -x4*u4, -y4*u4, 0 ],
            [ 0, 0, 0, x4, y4, 1, -x4*v4, -y4*v4, 0 ],
            [ 0, 0, 0,  0,  0, 0,      0,      0, 1 ],
            ])
    B = np.matrix([
            [ u1 ],
            [ v1 ],
            [ u2 ],
            [ v2 ],
            [ u3 ],
            [ v3 ],
            [ u4 ],
            [ v4 ],
            [ 1  ],
            ])
    
    X = A.I * B
    X.shape = (3, 3)
    return X.tolist()


#main
if __name__ == '__main__':
    #ファイル名を取得
    file_name = "../shiin/hand_landmark_10000.csv"
    #ファイルを1行ずつ読み込む
    with open(file_name, mode='r') as f:
        lines = f.readlines()
    #各行を読み込んで、座標を取得
    for k in range(0,len(lines),100):
        #最初の行は飛ばす
        if lines[k][0] == 't':
            continue
        print(k)
        lines[k] = lines[k].rstrip('\n')
        lines[k] = lines[k].split(',')
        label = int(lines[k][0])
        hand_xyz = [float(x) for x in lines[k][1:]]

        x_coords = hand_xyz[::3]
        y_coords = hand_xyz[1::3]
        z_coords = hand_xyz[2::3]

        #xを反転    
        x_coords = [1-x for x in x_coords] #反転している場合は必要

        #===================================================================================================
        
        #z = 0の平面に変換
        res = rotate_points_yaw_pitch(np.array([x_coords, y_coords, z_coords]).T)
        res = rotate_points_roll(res)
        x_coords, y_coords, z_coords = res.T

        cnt = 0
        x,y = 0,0
        for j in range(0,3):
            for i in range(0,3):
                # 8  7  6  5
                # 12 11 10 9
                # 16 15 14 13
                # 20 19 18 17
                index = 8+j*4-i
                src = [[x_coords[index], y_coords[index]], [x_coords[index-1], y_coords[index-1]], [x_coords[index+4], y_coords[index+4]], [x_coords[index+3], y_coords[index+3]]]
                # 0 1 2 3
                # 1
                # 2
                # 3
                dst = [ [i, j], [i+1, j], [i ,j+1], [i+1, j+1] ]
                X = find_homography(src, dst)
                xc = []
                yc = []
                #変換後の座標を計算
                src = np.array([x_coords[4], y_coords[4], 1])
                dst = np.dot(X, src)
                new_x_coords = dst[0]/dst[2]
                new_y_coords = dst[1]/dst[2]
                #枠内かチェック
                flg = (i == 0 or i <= new_x_coords) and (i+1 == 3 or new_x_coords <= i+1) and (j == 0 or j <= new_y_coords) and (j+1 == 3 or new_y_coords <= j+1)
                #print(new_x_coords, new_y_coords,flg,i,j,index)
                if flg:
                    x = new_x_coords
                    y = new_y_coords
                    cnt+=1
        if cnt == 1:
            #プロット。labelの値によって色を変える
            cm = plt.get_cmap("Spectral")
            plt.scatter(x, y, color=cm(label/10), s=10)


    x_coords = [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3]
    y_coords = [0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]
    plt.scatter(x_coords, y_coords, c = 'b')
    
    plt.xlim(-1, 4)
    plt.ylim(-1, 4)

    #y軸は下向きに正
    plt.gca().invert_yaxis()
    plt.show()


                
