#Arduinoからシリアル通信でデータを受信して、そのまま表示するプログラム
import serial
#シリアル通信でデータを受信する
if __name__ == '__main__':
    ser = serial.Serial('/dev/tty.usbmodem1301', 9600)
    while True:
        row = ser.readline() # 1行読み取り
        text = row.rstrip().decode('utf-8') # 末尾の改行コードを除去
        print("row:", row, "data:", text)
