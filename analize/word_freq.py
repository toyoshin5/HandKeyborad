import re
import matplotlib.pyplot as plt
import japanize_matplotlib

def convert_katakana_to_hiragana(char):
    katakana_to_hiragana = str.maketrans('ァィゥェォッャュョヮ', 'ぁぃぅぇぉっゃゅょわ')
    return char.translate(katakana_to_hiragana)

def remove_diacritics(char):
    # 濁音、半濁音、拗音を取り除く
    return re.sub('[\u3099\u309A\u309B\u309C]', '', char)

def count_kana_frequency(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    text = ''.join(remove_diacritics(convert_katakana_to_hiragana(char)) for char in text)

    hiragana_pattern = re.compile('[ぁ-ん]')

    hiragana_count = {}
    total_count = 0

    for char in re.findall(hiragana_pattern, text):
        total_count += 1
        if char in hiragana_count:
            hiragana_count[char] += 1
        else:
            hiragana_count[char] = 1

    # あ-んの順に並び替え
    hiragana_count = dict(sorted(hiragana_count.items(), key=lambda x: ord(x[0])))

    # 出現頻度を全体の出現回数で割る
    relative_frequency = {char: count / total_count for char, count in hiragana_count.items()}

    return relative_frequency
def plot_kana_frequency(hiragana_count, title):
    labels, values = zip(*hiragana_count.items())

    plt.figure(figsize=(10, 6))
    plt.bar(labels, values)
    plt.title(title)
    plt.xlabel('Hiragana Characters')
    plt.ylabel('Frequency')
    plt.show()
    
def plot_combined_kana_frequency(hiragana_count1, hiragana_count2):
    # 両方のデータセットで共通のひらがなのみを取り出す
    common_hiragana = sorted(set(hiragana_count1.keys()) & set(hiragana_count2.keys()), key=lambda x: ord(x))

    # 共通のひらがなに対する相対的な頻度を抽出
    values1 = [hiragana_count1[char] for char in common_hiragana]
    values2 = [hiragana_count2[char] for char in common_hiragana]

    width = 0.35
    x = range(len(common_hiragana))

    plt.figure(figsize=(15, 6))
    plt.bar(x, values1, width, label='File 1')
    plt.bar([i + width for i in x], values2, width, label='File 2')

    plt.title('Relative Hiragana Frequency Comparison (あ-ん order)')
    plt.xlabel('Hiragana Characters')
    plt.ylabel('Relative Frequency')
    plt.xticks([i + width/2 for i in x], common_hiragana)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    file_path1 = '../2ndExp/word_set.txt'
    file_path2 = ''  # 2つ目のファイルのパスを適切なものに変更する

    hiragana_count1 = count_kana_frequency(file_path1)

    # file_path2 が空でない場合にのみ処理を行う
    if file_path2:
        hiragana_count2 = count_kana_frequency(file_path2)
        plot_combined_kana_frequency(hiragana_count1, hiragana_count2)
    else:
        # file_path2 が空の場合、file_path1 のみのグラフを表示する
        plot_kana_frequency(hiragana_count1,"Hiragana Frequency")

