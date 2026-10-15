from sklearn.ensemble import RandomForestClassifier

X = [[1, 100], [2, 200], [10, 1000], [12, 1200], [3, 250]]
y = [1, 1, 0, 0, 1]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

print(model.predict([[5, 300]]))
