import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import load_data, preprocess_data
from model import IrisNN

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
iris_data = load_data(url)

X_test_tensor, y_test_tensor, train_loader = preprocess_data(iris_data)

model = IrisNN()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train the model

num_epochs = 100
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

with torch.no_grad():
    model.eval()
    test_outputs = model(X_test_tensor)
    _, predicted = torch.max(test_outputs, 1)
    accuracy = (predicted == y_test_tensor).float().mean()
    print(f'Accuracy on test set: {accuracy:.4f}')

# Visualizing the results

plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_test_tensor[:, 0], y=X_test_tensor[:, 1], hue=predicted.numpy(), palette='Set2', style=y_test_tensor.numpy(), markers=["o", "s", "D"])
plt.title('Iris Species Classification Results')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend(title='Predicted Species', loc='upper left')
plt.show() 