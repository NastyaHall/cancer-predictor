import pandas as pd 
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle 

current_directory = os.getcwd()
main_directory = os.path.abspath(os.path.join(current_directory, os.pardir))
data_directory = os.path.join(main_directory, 'data')

def create_model(data):
    x = data.drop(['diagnosis'], axis=1)
    y = data['diagnosis']

    # scale the data
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    # split the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # train 
    model = LogisticRegression()
    model.fit(x_train, y_train)

    # test model
    y_pred = model.predict(x_test)
    # print('Accuracy of the model: ', accuracy_score(y_test, y_pred))
    # print('Classification report: ', classification_report(y_test, y_pred))

    return model, scaler

def get_clean_data():
    data = pd.read_csv(f"{data_directory}/data.csv")
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    return data

def main():
    data = get_clean_data()
    model, scaler = create_model(data)

    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    

if __name__ == "__main__":
    main()