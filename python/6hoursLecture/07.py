'''
score_file = open("score.txt", "w", encoding="utf8")
print("수학 : 0 ", file=score_file)
print("영어 : 50 ", file=score_file)
score_file.close()

score_file = open("score.txt", "a", encoding="utf8")  #이어쓰기
score_file.write("과학 : 80")
score_file.write("\n코딩 : 25")
score_file.close()
'''

#한번에 다 읽어오는 것
score_file = open("score.txt", "r", encoding="utf8")
print(score_file.read())
score_file.close()


