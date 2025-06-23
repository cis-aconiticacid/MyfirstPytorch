import torch 
import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
def get_data_loader(is_train):
    to_tensor= transforms.Compose([transforms.ToTensor()])
    data_set = MNIST("", is_train, transform=to_tensor, download=True)
    return DataLoader(data_set, batch_size=13, shuffle=True)
def evaluate(test_data, net):
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for x, y in test_data:
            output = net(x)
            _, predicted = torch.max(output, 1)
            n_correct += (predicted == y).sum().item()
            n_total += y.size(0)
    return n_correct / n_total
def main():
    train_data = get_data_loader(is_train=True)
    test_data = get_data_loader(is_train=False)
    net = Net()
    
    print("Initial accuracy: ", evaluate(test_data, net) * 100)
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    
    for epoch in range(2):
        for x, y in train_data:
            optimizer.zero_grad()
            output = net(x)
            loss = loss_function(output, y)
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
        accuracy = evaluate(test_data, net)
        print(f"Test Accuracy after epoch {epoch+1}: {accuracy * 100:.2f}%")
def evaluate_model(net, test_data):
    accuracy = evaluate(test_data, net)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
if __name__ == "__main__":
    main()