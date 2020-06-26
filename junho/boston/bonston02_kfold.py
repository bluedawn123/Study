from keras.datasets import boston_housing

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

print(train_data.shape)   #(404, 13)
print(test_data.shape)    #(102, 13)
