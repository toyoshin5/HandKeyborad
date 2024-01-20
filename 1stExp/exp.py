
import mediapipe as mp
import numpy as np
import pandas as pd
import cv2
import serial
import time

VIDEOCAPTURE_NUM = 0 #ビデオキャプチャの番号
ARDUINO_PATH = "/dev/tty.usbmodem2101" #Arduinoのシリアルポート

class CharProvider:
    test_string = [
                  "あかさたなはまやらわあいうえおかきくけこ",
                  "あいうえおかきくけこさしすせそ小たちつてとなにぬねのはひふへほ小まみむめもやゆよらりるれろわをん小",
                  "あいうえおかきくけこさしすせそ小たちつてとなにぬねのはひふへほ小まみむめもやゆよらりるれろわをん小",
                  "あいうえおかきくけこさしすせそ小たちつてとなにぬねのはひふへほ小まみむめもやゆよらりるれろわをん小",
                  "また小なをわいよむつそれやしひちのへらけにこかみ小あお小ほんねるはりゆすときてふくさえうめろせもぬ",
                  "えちておむへやこふとようもめつそたまほせひさらわりね小いみあにか小ゆくるのすはんぬ小しなれきろをけ",
                  "ゆえらんねあしふてりとを小めむのまろこなおうもひるれやそたきよかち小にわ小ぬくほせすけいみはへさつ",
                  "あいうえおかきくけこさしすせそ小たちつてとなにぬねのはひふへほ小まみむめもやゆよらりるれろわをん小",
                  "あいうえおかきくけこさしすせそ小たちつてとなにぬねのはひふへほ小まみむめもやゆよらりるれろわをん小",
                  "あいうえおかきくけこさしすせそ小たちつてとなにぬねのはひふへほ小まみむめもやゆよらりるれろわをん小",
                  "てとへひおれもゆそちこきかわたつぬくはせ小にまんらしむすふえるけを小ろりうなのみいよ小めさあねほや",
                  "ゆえらんねあしふてりとを小めむのまろこなおうもひるれやそたきよかち小にわ小ぬくほせすけいみはへさつ",
                  "れてをやせひ小さくわそもりんみつむおしまにらへぬよ小とけふめたすかほちこきなねいゆうえの小はあるろ",
                  "ねにちらさきとをのみ小めてむゆ小やあせけすくれろぬうこわそほえひりよかいんおもへふるは小またなつし",
                  "ゆつくを小てぬまるあいへこきすねさかのにとやひほちらせなはりえけ小む小もめたうおんふろそわよしれみ",
                  "てとへひおれもゆそちこきかわたつぬくはせ小にまんらしむすふえるけを小ろりうなのみいよ小めさあねほや",
                  ]
    strset = 0
    index = 0
    word_mode = True
    df = pd.read_csv("../50on.csv",encoding="UTF-8", header=None)

    def __init__( self, word_mode = True ):
        self.word_mode = word_mode
    def get_char(self):
        c = self.test_string[self.strset][self.index]
        if c == "小":
            return "変換キー"
        return c
    def print_char(self):
        c = self.test_string[self.strset][self.index]
        print(c)
    def __init__(self):
        self.index = 0
    def next(self):
        self.index += 1
        if self.index >= len(self.test_string[self.strset]):
            self.index = 0
            self.strset += 1
            if self.strset >= len(self.test_string):
                #終了
                print("終了")
                exit()
            print(str(self.strset-1)+"/"+str(len(self.test_string)-1)+"セット終了")
            print("次の文字セットへ移ります")
            time.sleep(5)
    def prev(self):
            
        self.index -= 1
        if self.index < 0:
            self.index = 0
    def get_shiin(self):
        for i in range(len(self.df)):
            for j in range(len(self.df.iloc[i])):
                if self.df.iloc[i][j] == self.test_string[self.strset][self.index]:
                    return i
        return -1
    def get_boin(self):
        for i in range(len(self.df)):
            for j in range(len(self.df.iloc[i])):
                if self.df.iloc[i][j] == self.test_string[self.strset][self.index]:
                    return j
        return -1
    
    


def write_header_shiin(f):
    s = "target,"
    for i in range(21):
        s += ("x" + str(i) + ",y" + str(i) + "," + "z" + str(i) + ",")
    s = s[:-1]
    f.write(s + "\n")
    return

def write_header_boin(f):   
    s = "target,release_frame,shiin,duration,"
    for i in range(21):
        s += ("x" + str(i) + ",y" + str(i) + "," + "z" + str(i) + ",")
    s = s[:-1]
    f.write(s + "\n")
    return

#各landmarkのx,y座標をカンマ区切りでまとめる
#landmark:手の座標,target:ターゲットの文字の子音
def write_csv_shiin(f,landmark,target):
    s = target + ","
    for i in range(21):
        s += str(landmark[i].x) + "," + str(landmark[i].y) + "," + str(landmark[i].z) + ","
    s = s[:-1]
    f.write(s + "\n")
    return

#landmark:手の座標,target:ターゲットの文字の母音,release_frame:離してからのフレーム数,shiin:ターゲットの文字の子音
def write_csv_boin(f,landmark,target,release_frame,shiin,d):
    s = target + "," + str(int(release_frame)) + "," + shiin + "," + str(d) + ","
    for i in range(21):
        s += str(landmark[i].x) + "," + str(landmark[i].y) + "," + str(landmark[i].z) + ","
    s = s[:-1]
    f.write(s + "\n")
    return

def undo_csv_shiin(f):
    f.seek(0, 2)  # ファイル末尾に移動
    end_position = f.tell()  # ファイル末尾の位置を記録

    # ファイル末尾から1文字ずつ戻りながら改行文字を探す
    cnt = 2
    for pos in range(end_position - 1, -1, -1):
        f.seek(pos)
        if f.read(1) == '\n':
            cnt -= 1
        if cnt == 0:
            break
    # ファイルを最後の行の開始位置まで切り詰める
    f.truncate(pos)
    #改行する
    f.write("\n")
    return

def undo_csv_boin(f):
    f.seek(0, 2)  # ファイル末尾に移動
    end_position = f.tell()  # ファイル末尾の位置を記録

    # ファイル末尾から1文字ずつ戻りながら改行文字を探す
    cnt = 12
    for pos in range(end_position - 1, -1, -1):
        f.seek(pos)
        if f.read(1) == '\n':
            cnt -= 1
        if cnt == 0:
            break
    # ファイルを最後の行の開始位置まで切り詰める
    f.truncate(pos)
    #改行する
    f.write("\n")
    return

def landmark_in_view(hand_landmarks):
    for i in range(21):
        if hand_landmarks.landmark[i].x < 0 or hand_landmarks.landmark[i].x > 1:
            return False
        if hand_landmarks.landmark[i].y < 0 or hand_landmarks.landmark[i].y > 1:
            return False
    return True

#main


if __name__ == "__main__":
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.8,
    )
    mp_drawing = mp.solutions.drawing_utils
    #シリアル通信の設定
    ser = serial.Serial(ARDUINO_PATH, 9600)
    #ウェブカメラからの入力
    cap = cv2.VideoCapture(VIDEOCAPTURE_NUM)
    #csvファイルに書き込み
    f_s = open('hand_landmark_shiin.csv', 'a+')
    f_b = open('hand_landmark_boin.csv', 'a+')
    #ファイルが空の場合はヘッダーを書き込み
    if f_s.tell() == 0:
        write_header_shiin(f_s)
    if f_b.tell() == 0:
        write_header_boin(f_b)
    #文字提供
    cp = CharProvider()
    #現在手が画面内にあるかどうか
    hand_in_view = False
    #タップしているか
    is_tapping = False
    release_cnt = 0
    #スペースキーを押して開始
    print("Enterキーを押して開始")
    input()
    landmarks_release = []
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
        
        if results.multi_hand_landmarks and landmark_in_view(results.multi_hand_landmarks[0]):
            if not hand_in_view:
                print("\r"+str(cp.get_char()),"を入力してください　　　　　　　　　　",end="　",flush=True)
            hand_in_view = True
            #len(results.multi_hand_landmarks) = 写っている手の数
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS) 
            # 親指の先端は青色で描画
            cv2.circle(image, (int(hand_landmarks.landmark[4].x * image.shape[1]), int(hand_landmarks.landmark[4].y * image.shape[0])), 8, (255, 0, 0), -1)
            if ser.in_waiting > 0 and release_cnt == 0:
                row = ser.readline()
                msg = row.decode('utf-8').rstrip()
                if msg == "tap":
                    #離すところまで検出できてから保存処理をおこなう
                    landmark_tap = hand_landmarks.landmark
                    is_tapping = True
                if "release" in msg and is_tapping:
                    release_cnt = 1    
                    duration = float(msg.split(",")[1])
            if release_cnt >= 1:
                #離してから10フレームのデータを取得
                landmarks_release.append(hand_landmarks.landmark)
                release_cnt += 1
                if ser.in_waiting > 0:
                    _ = ser.readline()#捨て
            if release_cnt > 10:#離してから10フレーム経過したら
                #書き込み処理
                #csvファイルに書き込み(子音)
                target = cp.get_shiin()
                write_csv_shiin(f_s,landmark_tap,str(target))
                #csvファイルに書き込み(母音)
                target = cp.get_boin()
                write_csv_boin(f_b,landmark_tap,str(target),0,str(cp.get_shiin()),duration)
                for i in range(len(landmarks_release)):
                    #csvファイルに書き込み(母音)
                    write_csv_boin(f_b,landmarks_release[i],str(target),i+1,str(cp.get_shiin()),duration)
                #次の文字へ
                print("OK")
                cp.next()
                print(cp.get_char(),"を入力してください　　　　　　　　　　",end="　",flush=True)
                is_tapping = False
                release_cnt = 0
                landmarks_release = []
        else:
            if hand_in_view:

                print("\r手をカメラの画角内に収めてください",end="　",flush=True)
            hand_in_view = False
            is_tapping = False
            release_cnt = 0
            landmarks_release = []
            if ser.in_waiting > 0:
                _ = ser.readline()#捨て
            
        cv2.imshow('MediaPipe Hands', image)
        #キー入力
        key = cv2.waitKey(5)
        #スペースキーで戻る
        if key == ord(' '):
            cp.prev()
            print("\r1文字戻る　　　　　　　　　　　　　　　　　　　　　　　　　")
            undo_csv_shiin(f_s)
            undo_csv_boin(f_b)
            print(cp.get_char(),"を入力してください　　　　　　　　　　",end="　",flush=True)


    hands.close()
    cap.release()
