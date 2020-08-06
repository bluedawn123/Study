from keras.preprocessing.text import Tokenizer

text = "똥이라구 똥"

token = Tokenizer()
token.fit_on_texts([text])

print("token.word_index : ", token.word_index)

x = token.texts_to_sequences([text])
print("x : ", x)

from keras.utils import to_categorical

word_size = len(token.word_index) + 1
x = to_categorical(x, num_classes=word_size)
print("원핫인코딩 후 x : ", x)


