
# Description: 手のランドマークをcsvファイルに書き込むプログラム
import datetime
import mediapipe as mp
import cv2
import serial

target_dict = {0:"あ",1:"か",2:"さ",3:"た",4:"な",5:"は",6:"ま",7:"や",8:"ら",9:"わ",10:"小"}
rev_target_dict = {"あ":0,"か":1,"さ":2,"た":3,"な":4,"は":5,"ま":6,"や":7,"ら":8,"わ":9,"小":10}

MODE = "3D" #2D or 3D
ARDUINO_PATH = "/dev/tty.usbmodem1201" #Arduinoのシリアルポート

def write_shiin_header(f):
    s = "target,"
    for i in range(21):
        if MODE == "2D":
            s += ("x" + str(i) + ",y" + str(i) + ",")
        elif MODE == "3D":
            s += ("x" + str(i) + ",y" + str(i) + "," + "z" + str(i) + ",")
    s = s[:-1]
    f.write(s + "\n")
    return
#各landmarkのx,y座標をカンマ区切りでまとめる
def write_shiin_csv(f,landmark,target):
    s = target + ","
    for i in range(21):
        if MODE == "2D":
            s += str(landmark[i].x) + "," + str(landmark[i].y) + ","
        elif MODE == "3D":
            s += str(landmark[i].x) + "," + str(landmark[i].y) + "," + str(landmark[i].z) + ","
    s = s[:-1]
    f.write(s + "\n")
    return


#main
if __name__ == "__main__":
    #ターゲットの段の入力
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    mp_drawing = mp.solutions.drawing_utils
    #ウェブカメラからの入力
    cap = cv2.VideoCapture(0)
    #シリアル通信の設定
    ser = serial.Serial(ARDUINO_PATH, 9600)
    #現在日時
    now = datetime.datetime.now()
    now_str = now.strftime('%m%d_%H%M')
    #csvファイルに書き込み
    f_shiin = open('hand_landmark_shiin_'+now_str+'.csv', 'a')  
    f_boin = open('hand_landmark_boin_'+now_str+'.csv', 'a')
    #ファイルが空の場合はヘッダーを書き込み
    if f_shiin.tell() == 0:
        write_shiin_header(f_shiin)
    if f_boin.tell() == 0:
        write_shiin_header(f_boin)

    cnt = 0
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("カメラから映像を取得できませんでした")
            continue
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)#反転
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]#手の骨格を取得
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            if ser.in_waiting > 0:
                #シリアル通信で受け取った文字列をデコード
                row = ser.readline()
                msg = row.decode('utf-8').rstrip()
                if msg == "tap":
                    #TODO ： targetを決定して、csvに書き込み
                    write_shiin_csv(f_shiin,hand_landmarks.landmark,target)
            #if msg == "release":
                #TODO ： どのようなデータを取るかを考えて、csvに書き込み
            cnt += 1
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
        print("\r" + str(cnt),end="")

    hands.close()
    cap.release()
    f_shiin.close()
    f_boin.close()
