import cv2
class HiraganaImgManager:
    #ひらがなと画像の辞書
    hiragana_img_dict = {}
    def __init__(self):
        #ひらがなの画像を読み込んで辞書に格納
        for char in ["あ","い","う","え","お","か","き","く","け","こ","さ","し","す","せ","そ","た","ち","つ","て","と","な","に","ぬ","ね","の","は","ひ","ふ","へ","ほ","ま","み","む","め","も","や","ゆ","よ","ら","り","る","れ","ろ","わ","を","ん","だ"]:
            self.hiragana_img_dict[char] = cv2.imread("image/"+char+".png", cv2.IMREAD_UNCHANGED)
    #画像の任意の位置にひらがなを貼り付ける
    def putHiragana(self, char, img, pos, size):
         # bgraに変換
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        # ひらがなの画像を取得
        hiragana_img = self.hiragana_img_dict[char]
        # ひらがな画像のサイズを変更
        hiragana_img = cv2.resize(hiragana_img, (size, size))
        
        # ひらがな画像の高さと幅を取得
        h, w = hiragana_img.shape[:2]

        # 貼り付け先領域の座標を計算
        y1, y2 = pos[0], pos[0] + h
        x1, x2 = pos[1], pos[1] + w
        
        # 貼り付け先領域のアルファチャンネルを計算
        alpha_s = hiragana_img[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        
        # 画像を重ねる
        for c in range(0, 3):
            img[y1:y2, x1:x2, c] = (alpha_s * hiragana_img[:, :, c] + alpha_l * img[y1:y2, x1:x2, c])
            
        # bgrに戻す
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        return img






        

    
        
