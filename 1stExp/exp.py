
import mediapipe as mp
import numpy as np
import pandas as pd
import cv2
import serial

VIDEOCAPTURE_NUM = 0 #ビデオキャプチャの番号
ARDUINO_PATH = "/dev/tty.usbmodem1201" #Arduinoのシリアルポート

class CharProvider:
    testString = "あいうえおかきくけこさしすせそたちつてとなにぬねの"
    index = 0
    f = pd.read_csv("../50on.csv",encoding="UTF-8", header=None)
    def print_char(self):
        c = self.testString[self.index]
        print(c)
    def __init__(self):
        self.index = 0
        self.print_char()
    def next(self):
        self.index += 1
        if self.index >= len(self.testString):
            self.index = 0
        self.print_char()
    def prev(self):
        self.index -= 1
        if self.index < 0:
            self.index = len(self.testString) - 1
        self.print_char()
    def get_shiin(self):
        for i in range(len(f)):
            for j in range(len(f.iloc[i])):
                if f.iloc[i][j] == self.testString[self.index]:
                    return i
        return -1
    def get_boin(self):
        for i in range(len(f)):
            for j in range(len(f.iloc[i])):
                if f.iloc[i][j] == self.testString[self.index]:
                    return j
        return -1
    
    


def write_header(f):
    s = "target,"
    for i in range(21):
        s += ("x" + str(i) + ",y" + str(i) + "," + "z" + str(i) + ",")
    s = s[:-1]
    f.write(s + "\n")
    return
#各landmarkのx,y座標をカンマ区切りでまとめる
def write_csv(f,landmark,target):
    s = target + ","
    for i in range(21):
        s += str(landmark[i].x) + "," + str(landmark[i].y) + "," + str(landmark[i].z) + ","
    s = s[:-1]
    f.write(s + "\n")
    return

def shiin_from_kana(f,kana):
    #fはpdのデータフレーム
    #kanaはひらがな
    #fの中からkanaに対応する行を探して返す
    for i in range(len(f)):
        for j in range(len(f.iloc[i])):
            if f.iloc[i][j] == kana:
                return i
    return 
def boin_from_kana(f,kana):
    #fはpdのデータフレーム
    #kanaはひらがな
    #fの中からkanaに対応する行を探して返す
    for i in range(len(f)):
        for j in range(len(f.iloc[i])):
            if f.iloc[i][j] == kana:
                return j
    return 
#main
testString = "あいうえおかきくけこさしすせそたちつてとなにぬねの"

if __name__ == "__main__":
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.3,
    )
    mp_drawing = mp.solutions.drawing_utils
    #シリアル通信の設定
    #ser = serial.Serial(ARDUINO_PATH, 9600)
    #ウェブカメラからの入力
    cap = cv2.VideoCapture(VIDEOCAPTURE_NUM)
    #csvファイルに書き込み
    f = open('hand_landmark_shiin.csv', 'a')  
    #ファイルが空の場合はヘッダーを書き込み
    if f.tell() == 0:
        write_header(f)
    #文字提供
    cp = CharProvider()
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
            if ser.in_waiting > 0:
                row = ser.readline()
                msg = row.decode('utf-8').rstrip()
                if msg == "tap":
                    #csvファイルに書き込み(子音)
                    target = cp.get_shiin()
                    write_csv(f,hand_landmarks.landmark,target)
                    #次の文字へ
                    cp.next()
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27: #ESCキーで終了
            break


    hands.close()
    cap.release()
