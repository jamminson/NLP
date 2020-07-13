from sklearn.model_selection import KFold

X = [1, 2, 3, 4, 5, 6, 7, 8]
Y = [[1, 2, 3, 4, 5, 6, 7, 8], ['a', 'b', 'c', 'd', 'e']]

kf = KFold(n_splits=3)
print(kf.get_n_splits(X))
for train_index in kf.split(X):
    print(train_index)

for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)

print("Here")

print((600 - 6) % 9)

k = 4
folds = list()
for _ in range(k):
    folds.append(X[(_ * 2):(_ * 2 + 2)])

# folds.append(X[len(X)-2:])

print(folds)

for _ in range(k):
    x_cv = list()
    for i in folds[:_]:
        x_cv.append(i)

    for i in folds[_ + 1:]:
        x_cv.append(i)

    print(x_cv)

for i in range(len(Y)):
    Y[i] = list(zip(Y[i][:-1], Y[i][1:]))

print(Y)