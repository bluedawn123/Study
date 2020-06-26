
import requests
import zipfile
from io import StringIO
import io
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# url
mush_data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
s = requests.get(mush_data_url).content

#데이터형식
mush_data = pd.read_csv(io.StringIO(s.decode("utf-8")), header=None)


mush_data.columns = ["classes", "cap_shape", "cap_surface",
                     "cap_color", "odor", "bruises",
                     "gill_attachment", "gill_spacing",
                     "gill_size", "gill_color", "stalk_shape",
                     "stalk_root", "stalk_surface_above_ring",
                     "stalk_surface_below_ring",
                     "stalk_color_above_ring",
                     "stalk_color_below_ring",
                     "veil_type", "veil_color","ring_number",
                     "ring_type", "spore_print_color",
                     "population", "habitat"]


mush_data_dummy = pd.get_dummies(
    mush_data[["gill_color", "gill_attachment", "odor", "cap_color"]])


mush_data_dummy["flg"] = mush_data["classes"].map(
    lambda x: 1 if x == "p" else 0)


X = mush_data_dummy.drop("flg", axis=1)
Y = mush_data_dummy["flg"]


train_X, test_X, train_y, test_y = train_test_split(X,Y, random_state=42)


from sklearn.neighbors import KNeighborsClassifier

#모델구축
model = KNeighborsClassifier()

#모델학습
model.fit(train_X, train_y)

#정확도
print(model.score(test_X, test_y))