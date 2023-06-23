import pyautogui
import pyperclip
import pandas as pd

#文字入力中はマウスカーソルを動かさないようにする
class TextInputManager:
    prev_char = ""
    df = None
    #コンストラクタ
    def __init__(self):
        pyautogui.PAUSE = 0.01
        #CSVファイルを読み込む
        self.df = pd.read_csv("textinput/dakuon_rule.csv",encoding="UTF-8", names=[0,1,2],header=None)
    
        
    #macOSが実際に入力する関数
    def __mojiwrite(self,char):
        pyperclip.copy(char)
        pyautogui.hotkey('command', 'v')
    #キー入力を処理する関数
    def mojitype(self,char):
        if char == "小":
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
            if len(prevrow) == 0:
                #prev_charが見つからなかった場合、変換出来ない
                return
            #prev_charの行の中で、prev_charの次の文字を探す
            for i in range(len(prevrow)):
                if prevrow[i] == self.prev_char:
                    #次の文字を取得
                    char = prevrow[(i + 1) % len(prevrow)]
                    #一文字削除
                    pyautogui.press('backspace')
                    break
            if char == "小":
                return 
        #macOSに入力
        self.__mojiwrite(char)
        #prev_charを更新
        self.prev_char = char
