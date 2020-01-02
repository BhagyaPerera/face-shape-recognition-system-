import pickle


train_data=pickle.load(open('data.pickle','rb'))
train_target=pickle.load(open('target.pickle','rb'))
#loading arrays saved in last code

print(train_data.shape)
print(train_target.shape)


print(train_data.shape)


from sklearn.neighbors import KNeighborsClassifier



clsfr=KNeighborsClassifier()
clsfr.fit(train_data,train_target)

#training the KNN


import joblib

joblib.dump(clsfr,'KNN_model.sav')
#save the train machine learning model
