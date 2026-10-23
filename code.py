import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 1. Load and prepare the dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (crucial for neural networks)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert arrays to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).view(-1, 1) # Reshape for BCELoss
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)

# 2. Define the Neural Network model
class BreastCancerModel(nn.Module):
    def __init__(self, input_dim):
        super(BreastCancerModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, 16)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(16, 1)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        return out

input_dim = X_train.shape[1]
model = BreastCancerModel(input_dim)

# 3. Define loss and optimizer
criterion = nn.BCEWithLogitsLoss() # Combines Sigmoid + Binary Cross Entropy for stability
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 4. Training loop
epochs = 100
for epoch in range(epochs):
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 5. Evaluation
model.eval() # Set model to evaluation mode
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    # Apply sigmoid to get probabilities, then round to get binary predictions (0 or 1)
    predictions = torch.round(torch.sigmoid(test_outputs))
    
    # Calculate accuracy
    acc = accuracy_score(y_test_tensor.numpy(), predictions.numpy())
    print(f'\nTest Accuracy: {acc * 100:.2f}%')
