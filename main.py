import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import load_data, preprocess_data
from model import IrisNN

species_mapping = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}


def main():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    iris_data = load_data(url)

    X_test_tensor, y_test_tensor, train_loader = preprocess_data(iris_data)

    model = IrisNN(input_size=4, hidden_sizes=[8, 16], output_size=3, dropout_rate=0.3)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 100
    train_model(model, train_loader, criterion, optimizer, num_epochs)

    evaluate_model(model, X_test_tensor, y_test_tensor)


def train_model(model, train_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")


def evaluate_model(model, X_test_tensor, y_test_tensor):
    with torch.no_grad():
        model.eval()
        test_outputs = model(X_test_tensor)
        _, predicted = torch.max(test_outputs, 1)
        accuracy = (predicted == y_test_tensor).float().mean()
        print(f"Accuracy on test set: {accuracy:.4f}")

    visualize_results(X_test_tensor, predicted, y_test_tensor)


def visualize_results(X_test_tensor, predicted, y_test_tensor):
    predicted_species = [species_mapping[label.item()] for label in predicted]
    actual_species = [species_mapping[label.item()] for label in y_test_tensor]

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=X_test_tensor[:, 0].numpy(),
        y=X_test_tensor[:, 1].numpy(),
        hue=predicted_species,
        palette="Set2",
        style=actual_species,
        markers=["o", "s", "D"],
        legend="full",
    )
    plt.title("Iris Species Classification Results")
    plt.xlabel("Sepal Length")
    plt.ylabel("Sepal Width")
    plt.legend(title="Predicted Species", loc="upper left")
    plt.show()


if __name__ == "__main__":
    main()
