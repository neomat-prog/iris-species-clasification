import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset

# Load the Iris Dataset

def load_data(url):
    columns = ["SepalLength", "SepalWidth","PetalLength", "PetalWidth", "Species"]
    iris_data = pd.read_csv(url, names=columns)
    return iris_data

def preprocess_data(iris_data):
    # Encode species as numbers
    label_encoder = LabelEncoder()
    iris_data['Species'] = label_encoder.fit_transform(iris_data['Species'])

    X = iris_data.drop('Species', axis=1).values
    y = iris_data['Species'].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    return X_test_tensor, y_test_tensor, train_loader