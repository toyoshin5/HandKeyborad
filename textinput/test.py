from text_input_manager import TextInputManager
while True:
    text_mng = TextInputManager()
    inputstr = "ひ小小かひ小小かのらんと小せる"
    for i in range(len(inputstr)):
        text_mng.mojitype(inputstr[i])