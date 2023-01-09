import torch
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the data from the CSV file
df = pd.read_csv('./data/cleaned_data.csv')


# Select the relevant columns and split into features and labels
features = df.drop(columns=["rating", "game_title"])
num_features = features.shape[1]

labels = df["rating"]

# One-hot encode the non-binary categorical columns
# categorical_columns = ["game_title", "release_date", "developer", "publisher"]
# features = pd.get_dummies(features, columns=categorical_columns)

# Convert the data to tensors
X = torch.tensor(features.values.astype(float), dtype=torch.float)

y = torch.tensor(labels.values, dtype=torch.float)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Define the model
class GamesRecommendationModel(torch.nn.Module):
    def __init__(self):
        super(GamesRecommendationModel, self).__init__()
        self.fc1 = torch.nn.Linear(in_features=num_features, out_features=64)
        self.fc2 = torch.nn.Linear(in_features=num_features, out_features=32)
        self.fc3 = torch.nn.Linear(in_features=num_features, out_features=1)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the model and define the loss and optimizer
model = GamesRecommendationModel()
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    # Forward pass
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)

    # Backward pass
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Evaluate the model on the testing data
y_pred = model(X_test)
test_loss = loss_fn(y_pred, y_test)
print("Test loss:", test_loss.item())
