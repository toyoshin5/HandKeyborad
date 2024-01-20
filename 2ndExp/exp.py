

import math
import sys
import mediapipe as mp
import numpy as np
import pandas as pd
import cv2
import serial
import joblib

VIDEOCAPTURE_NUM = 1 #ビデオキャプチャの番号
ARDUINO_PATH = "/dev/tty.usbmodem2101" #Arduinoのシリアルポート
sys.path.append("../image")
from hiragana_img_manager import HiraganaImgManager


class HiraganaUtil:
    target_dict = {0:"あ",1:"か",2:"さ",3:"た",4:"な",5:"は",6:"ま",7:"や",8:"ら",9:"小",10:"わ"}
    rev_target_dict = {"あ":0,"か":1,"さ":2,"た":3,"な":4,"は":5,"ま":6,"や":7,"ら":8,"小":9,"わ":10}
    df = pd.read_csv("../50on.csv",encoding="UTF-8", header=None)
    def coord_to_char(self,x,y):
        return self.df.iloc[x][y]
    def num_to_shiin(self,num):
        return self.target_dict[num]
    def shiin_to_num(self,shiin):
        return self.rev_target_dict[shiin]
    
    def get_shiin(self,chr):
        for i in range(len(self.df)):
            for j in range(len(self.df.iloc[i])):
                if self.df.iloc[i][j] == chr:
                    return i
        return -1
    def get_boin(self,chr):
        for i in range(len(self.df)):
            for j in range(len(self.df.iloc[i])):  
                if self.df.iloc[i][j] == chr:
                    return j
        return -1
    
class CharManager:
    henkan_df = None
    prectice_words = ["あお","かき","ねこ","わたし","ありがとう","いちご","おはよう","さようなら","さくら","すし"]
    exp_words = ["あお","かき","ねこ","わたし","ありがとう","いちご","おはよう","さようなら","さくら","すし","あか","あかい","あかるい","あき","あく","あける","あげる","あさ","あさごはん","あさって","あし","あした","あそぶ","あたたかい","あたま","あたり"]
    word_list = []
    word = ""
    index = 0
    def __init__(self,is_practice):
        if is_practice:
            self.word_list = self.prectice_words
        else:
            self.word_list = self.exp_words
        self.index = 0
        self.henkan_df = pd.read_csv("textinput/dakuon_rule.csv",encoding="UTF-8", names=[0,1,2],header=None)
    def next_word(self):
        self.index += 1
        self.word = ""
        if self.index >= len(self.word_list):
            self.index = 0
    def delete(self):
        if len(self.word) > 0:
            self.word = self.word[:-1]
    def input(self,char):
        if char == "小" and len(self.word) > 0:
            #dfから、prev_charのセルを探す
            prevrow = []
            for index, r in self.df.iterrows():
                for col_label, cell_value in r.items():
                    # セルの値と比較して一致する場合
                    if cell_value == self.prev_char:
                        #その行を配列として取得
                        prevrow = r.values.tolist()
                        prevrow = [x for x in prevrow if str(x) != 'nan']
                        break
            # if len(prevrow) == 0:
            #     #prev_charが見つからなかった場合、変換出来ない
            #     return
            #prev_charの行の中で、prev_charの次の文字を探す
            for i in range(len(prevrow)):
                if prevrow[i] == self.prev_char:
                    #次の文字を取得
                    char = prevrow[(i + 1) % len(prevrow)]
                    #一文字削除
                    self.word = self.word[:-1]
                    break
        if char!="小":
            #入力に成功している場合
            self.word += char
            print(self.word)
            if self.word == self.word_list[self.index]:
                self.next_word()
        
    
    
def landmark_in_view(hand_landmarks):
    for i in range(21):
        if hand_landmarks.landmark[i].x < 0 or hand_landmarks.landmark[i].x > 1:
            return False
        if hand_landmarks.landmark[i].y < 0 or hand_landmarks.landmark[i].y > 1:
            return False
    return True


class HiraganaPredictor:
    shiinModelPath = "../analize/shiin_knn.pkl"
    boinModelPath = "../analize/boin_knn.pkl"
    average_of_hand = [0.7050574771237412, 0.7453036618739438, 4.220176619447094e-07, 0.6380715746489578, 0.5537182858377383, -0.04065226526935274, 0.5434539755573684, 0.4508825126798234, -0.03390088839189952, 0.44519127578768014, 0.4578004294824783, -0.024573237044675202, 0.37307509774826975, 0.4911852523618461, -0.012327942991832154, 0.5059959732602433, 0.4289161944964497, 0.05602689533240021, 0.4091820318843212, 0.3961846968947064, 0.0656691299043447, 0.34666060758071693, 0.3974543131953953, 0.05622074677117737, 0.2934040388063881, 0.40084353620359053, 0.04432528910008988, 0.4875388513352965, 0.5258393738025576, 0.06888396954525841, 0.3824632530048862, 0.5001935499669995, 0.08041936595391437, 0.31294162611955123, 0.49503579424764727, 0.05768020218652304, 0.254272694985392, 0.4891538744781836, 0.03621396545367038, 0.48339028725525146, 0.620041376293474, 0.07113007325131464, 0.3823291003992248, 0.598686353879096, 0.06993089157887986, 0.3179530892205949, 0.5856796870372836, 0.04052457782898027, 0.2617927516940426, 0.5772903134219355, 0.01728786701674265, 0.48810099663271433, 0.7192457505801241, 0.0684737746956558, 0.4057268740265577, 0.7043628966575376, 0.060886312493620955, 0.3526093629425231, 0.694492256150286, 0.04308659675822081, 0.30303866852684236, 0.6795648798394549, 0.027648239867118108]
    shiinClf = None
    boinClf = None
    #init
    def __init__(self):
        self.shiinClf = joblib.load(self.shiinModelPath)
        self.boinClf = joblib.load(self.boinModelPath)

    def __get_sam_coords(self,landmarks):
        coords = self.__landmarksToCoords(landmarks)
        #入力の手を回転
        coords = self.__rotate_points_yaw_pitch(coords)
        coords = self.__rotate_points_roll(coords)
        x_coords, y_coords, _ = coords.T
        #平均の手の回転
        ave_x_coords = self.average_of_hand[::3]
        ave_y_coords = self.average_of_hand[1::3]    
        ave_z_coords = self.average_of_hand[2::3]
        coords = self.__rotate_points_yaw_pitch(np.array([ave_x_coords, ave_y_coords, ave_z_coords]).T)
        coords = self.__rotate_points_roll(coords)
        ave_x_coords, ave_y_coords, ave_z_coords = coords.T
        #ホモグラフィー変換
        try:
            sam_x_coord,sam_y_coord = self.__homographyTransform(x_coords,y_coords,ave_x_coords, ave_y_coords)
        except:
            return 0,0
        return sam_x_coord,sam_y_coord
    def predict_shiin(self,landmarks):
        sam_x_coord,sam_y_coord = self.__get_sam_coords(landmarks)
        if sam_x_coord > 0.5 and sam_y_coord < 0.2:
            return sam_x_coord,sam_y_coord,100
        if sam_x_coord == 0 and sam_y_coord == 0:
            return sam_x_coord,sam_y_coord,-1
        Z = self.shiinClf.predict([[sam_x_coord,sam_y_coord]]) 
        return sam_x_coord,sam_y_coord,Z[0]
        
    #子音の判定
    def predict_boin(self,landmarks_tap,landmarks):
        if landmarks_tap is None or not landmarks_tap:
            return -1
        sam_x_coord,sam_y_coord = self.__get_sam_coords(landmarks)
        if sam_x_coord == 0 and sam_y_coord == 0:
            return -1
        sam_x_coord_tap,sam_y_coord_tap = self.__get_sam_coords(landmarks_tap)
        if sam_x_coord_tap == 0 and sam_y_coord_tap == 0:
            return -1
        #親指の位置の差分を計算
        diff_x = sam_x_coord - sam_x_coord_tap
        diff_y = sam_y_coord - sam_y_coord_tap
        #親指の位置の差分から、子音を計算
        Z = self.boinClf.predict([[diff_x,diff_y]])
        return Z[0]
        
    #4つの点から射影変換行列を求める関数
    def __find_homography(self,src, dst):
        
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
    def __in_rect(self,rect,target,i,j):
        #a - d
        #| e |
        #b - c
        a = (rect[0][0], rect[0][1])
        b = (rect[1][0], rect[1][1])
        c = (rect[2][0], rect[2][1])
        d = (rect[3][0], rect[3][1])
        e = (target[0], target[1])

        # 原点から点へのベクトルを求める
        vector_a = np.array(a)
        vector_b = np.array(b)
        vector_c = np.array(c)
        vector_d = np.array(d)
        vector_e = np.array(e)

        # 点から点へのベクトルを求める
        vector_ab = vector_b - vector_a
        vector_ae = vector_e - vector_a
        vector_bc = vector_c - vector_b
        vector_be = vector_e - vector_b
        vector_cd = vector_d - vector_c
        vector_ce = vector_e - vector_c
        vector_da = vector_a - vector_d
        vector_de = vector_e - vector_d

        # 外積を求める
        vector_cross_ab_ae = np.cross(vector_ab, vector_ae)
        vector_cross_bc_be = np.cross(vector_bc, vector_be)
        vector_cross_cd_ce = np.cross(vector_cd, vector_ce)
        vector_cross_da_de = np.cross(vector_da, vector_de)

        #普通であれば、全てマイナスであれば、点は四角形の内側にある
        #端っこの四角形の場合は、端方向であればはみ出てもいいということにする
        return (i == 0 or vector_cross_ab_ae < 0) and (j+1 == 3 or vector_cross_bc_be < 0) and (i+1 == 3 or vector_cross_cd_ce < 0) and (j == 0 or vector_cross_da_de < 0)

    def __homographyTransform(self,x_coords,y_coords,ave_x_coords, ave_y_coords):
            #各行を読み込んで、座標を取得
        for j in range(0,3):
            for i in range(0,3):
                # 8  7  6  5
                # 12 11 10 9
                # 16 15 14 13
                # 20 19 18 17
                index = 8+j*4-i
                rect = [[x_coords[index], y_coords[index]], [x_coords[index+4], y_coords[index+4]], [x_coords[index+3], y_coords[index+3]],[x_coords[index-1], y_coords[index-1]]]
                # 0 1 2 3
                # 1
                # 2
                # 3

                if self.__in_rect(rect, [x_coords[4],y_coords[4]],i,j):
                    dst = [[ave_x_coords[index], ave_y_coords[index]], [ave_x_coords[index+4], ave_y_coords[index+4]], [ave_x_coords[index+3], ave_y_coords[index+3]],[ave_x_coords[index-1], ave_y_coords[index-1]]]
                    A = self.__find_homography(rect, dst)
                    xc = []
                    yc = []
                    #変換後の座標を計算
                    src = np.array([x_coords[4], y_coords[4], 1])
                    dst = np.dot(A, src)
                    new_x_coord = dst[0]/dst[2]
                    new_y_coord = dst[1]/dst[2]
                    return new_x_coord, new_y_coord
    def __landmarksToCoords(self,landmarks):
        x = []
        y = []
        z = []
        for landmark in landmarks:
            x.append(landmark.x)
            y.append(landmark.y)
            z.append(landmark.z)
        return np.array([x,y,z]).T
    def __calc_regression_plane(self,coords):
        # 最小二乗法による平面の方程式の解を求める
        A = np.column_stack((coords[:, 0], coords[:, 1], np.ones(len(coords))))
        b = coords[:, 2]
        coefficients, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        # 平面の方程式の係数を取得
        a, b ,c = coefficients
        return a, b, c
    def __rotate_points_yaw_pitch(self,coords):
        #  指の平面を求める
        #5~20の点
        finger_coords = coords[5:21]
        # 最小二乗法による平面の方程式の解を求める
        a,b,_ = self.__calc_regression_plane(finger_coords)
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
    def __rotate_points_roll(self,coords):
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
    
class ThumbHistory:
    location_history = []
    shiin_history = []
    length = 10
    def __init__(self):
        self.location_history = []
        self.shiin_history = []
        for i in range(self.length):
            self.location_history.append([i*1000,i*1000])
            self.shiin_history.append(i)#適当な値を入れておく
    
    def frame_update(self,coord:[int,int],shiin:int):
        self.location_history.pop(0)
        self.shiin_history.pop(0)
        self.location_history.append(coord)
        self.shiin_history.append(shiin)
    
    def reset(self):
        self.location_history = []
        for i in range(self.length):
            self.location_history.append([0,0])
        
    def checkStop(self,frames):
        # #二次元データの分散を計算
        # var = np.var(self.location_history,axis=0)
        # var = np.sqrt(var[0]**2 + var[1]**2)
        # #分散が小さければ、停止していると判定
        # print(var)
        d = np.array(self.location_history[0]) - np.array(self.location_history[1]) 
        isStop = True
        for i in range(min(len(self.location_history)-1,frames)):
            #一つ前からの変化量を計算
            diff = np.array(self.location_history[i+1]) - np.array(self.location_history[i]) 
            #一つ前からの変化量が小さければ、停止していると判定
            if np.sqrt(diff[0]**2 + diff[1]**2) > 0.02:
                isStop = False
        # 子音に変化がなければ、停止していると判c定
        if isStop and len(set(self.shiin_history)) == 1:
            return True
        return False
#入力の状態の列挙型(入力前,入力中,入力後)
class InputState:
    BEFORE_INPUT = 0
    INPUTING = 1
    AFTER_INPUT = 2


#main
if __name__ == "__main__":
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.8,
    )
    mp_drawing = mp.solutions.drawing_utils
    #シリアル通信の設定
    #ser = serial.Serial(ARDUINO_PATH, 9600)
    #ウェブカメラからの入力
    cap = cv2.VideoCapture(VIDEOCAPTURE_NUM)
    #現在手が画面内にあるかどうか
    hand_in_view = False
 
    input_state = InputState.BEFORE_INPUT #入力の状態
    release_cnt = 0 #リリースされてからのフレーム数
    landmark_tap = None #タップされた時の手の座標
    pred_boin = None #推論された母音
    pred_shiin = 6 #推論された子音
    pred_shiin_tap = None #タップされた時の推論された子音
    thumb_ref_x_tap = 0 #タップされた時の親指の座標
    thumb_ref_y_tap = 0 #タップされた時の親指の座標
    #親指の履歴
    history = ThumbHistory()   
    #推論機
    hp = HiraganaPredictor()
    #文字系Util
    hu = HiraganaUtil()
    #ひらがな描画
    im = HiraganaImgManager()
    #スペースキーを押して開始
    print("Enterキーを押して開始")
    input()
    landmarks_release = []
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.multi_hand_landmarks and landmark_in_view(results.multi_hand_landmarks[0]):
            #手が写ったら
            if not hand_in_view:
                print("                               ",end="　",flush=True)
            
            hand_in_view = True
            hand_landmarks = results.multi_hand_landmarks[0]
            #======================
            #描画処理
            #======================

            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS) 
            # 親指の先端は青色で描画
            cv2.circle(image, (int(hand_landmarks.landmark[4].x * image.shape[1]), int(hand_landmarks.landmark[4].y * image.shape[0])), 8, (255, 0, 0), -1)
            #親指の座標
            thumb_x = int(hand_landmarks.landmark[4].x * image.shape[1])
            thumb_y = int(hand_landmarks.landmark[4].y * image.shape[0])
            #子音の判定
            thumb_ref_x,thumb_ref_y,pred_shiin = hp.predict_shiin(hand_landmarks.landmark)
            #入力前なら
            if input_state == InputState.BEFORE_INPUT:
                if pred_shiin != -1 and pred_shiin != 100:
                    #親指の位置にひらがなを描画
                    image = im.putHiragana(hu.coord_to_char(pred_shiin,0),image,[thumb_x,thumb_y],150,alpha=0.4)
                    landmark_tap = hand_landmarks.landmark
                    pred_shiin_tap = pred_shiin
                    thumb_ref_x_tap = thumb_ref_x
                    thumb_ref_y_tap = thumb_ref_y

            #入力中なら
            if input_state == InputState.INPUTING:
                if pred_shiin_tap != None and pred_shiin_tap != -1 and pred_shiin_tap != 100:
                    #親指の位置の周りに４つのひらがなを描画
                    for i in [0,1,2,3,4]:
                        hiragana = hu.coord_to_char(pred_shiin_tap,i)
                        if hiragana != '*':
                            image = im.putHiragana(hiragana,image,[thumb_x,thumb_y],150,alpha=0.5)
            #入力後なら
            if input_state == InputState.AFTER_INPUT:
                if pred_shiin_tap != None and pred_shiin_tap != -1 and pred_shiin_tap != 100 and pred_boin != None and pred_boin != -1:
                    #親指の位置の周りにひらがなを描画
                    kana = hu.coord_to_char(pred_shiin_tap,pred_boin)
                    if kana == '*':
                        kana = hu.coord_to_char(pred_shiin_tap,0)
                    image = im.putHiragana(kana,image,[thumb_x,thumb_y],150,alpha=1)
            #======================
            #状態遷移
            #======================
            if pred_shiin == 100:
                #親指が上に出たら、入力前に戻る
                input_state = InputState.BEFORE_INPUT
                release_cnt = 0
                landmarks_release = []
                history.reset()
            elif input_state == InputState.BEFORE_INPUT:
                history.frame_update([thumb_ref_x,thumb_ref_y],pred_shiin)
                if history.checkStop(3):
                    input_state = InputState.INPUTING
                    history.reset()
            if input_state == InputState.INPUTING:
                history.frame_update([thumb_ref_x,thumb_ref_y],pred_shiin)
                isMove = False
                #landmark_tap,hand_landmarks.landmarkの距離が0.025以上なら、移動していると判定
                if landmark_tap is not None:
                    if np.sqrt((thumb_ref_x_tap - thumb_ref_x)**2 + (thumb_ref_y_tap - thumb_ref_y)**2) > 0.05:
                        isMove = True
                if history.checkStop(20) or isMove:
                    pred_boin = hp.predict_boin(landmark_tap,hand_landmarks.landmark)
                    kana = hu.coord_to_char(pred_shiin_tap,pred_boin)
                    if kana == '*':
                        kana = hu.coord_to_char(pred_shiin_tap,0)
                    print(kana)
                    input_state = InputState.AFTER_INPUT
                    history.reset()
            
            if input_state == InputState.AFTER_INPUT:
                #離してから15フレームの間は表示する
                release_cnt += 1
                if release_cnt > 15:
                    input_state = InputState.BEFORE_INPUT
                    release_cnt = 0
                    landmarks_release = []
        else:
            if hand_in_view:
                print("\r手をカメラの画角内に収めてください",end="　",flush=True)
            hand_in_view = False
            input_state = InputState.BEFORE_INPUT
            release_cnt = 0
            landmarks_release = []
            # if ser.in_waiting > 0:
            #     _ = ser.readline()#捨て
            
        cv2.imshow('MediaPipe Hands', image)
        #キー入力
        key = cv2.waitKey(5)
        #スペースキーで戻る
        if key == ord(' '):
            hp.prev()
            print("\r1文字戻る　　　　　　　　　　　　　　　　　　　　　　　　　")
            # undo_csv_shiin(f_s)
            # undo_csv_boin(f_b)
            print(hp.get_char(),"を入力してください　　　　　　　　　　",end="　",flush=True)


    hands.close()
    cap.release()
