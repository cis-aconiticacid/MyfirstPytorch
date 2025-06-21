import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fac1 = torch.nn.Linear(28*28, 64)
        self.fac2 = torch.nn.Linear(64,64)
        self.fac3 = torch.nn.Linear(64,64)
        self.fac4 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fac1(x))
        x = torch.nn.functional.relu(self.fac2(x))
        x = torch.nn.functional.relu(self.fac3(x))
        x = torch.nn.functional.log_softmax(self.fac4(x), dim=1)
        return x

def get_data_loader(is_train):
    to_tensor = transforms.Compose([transforms.ToTensor()])
    data_set = MNIST("", is_train, transform=to_tensor, download=True)
    return DataLoader(data_set, batch_size=15, shuffle=True)

def evaluate(test_data, net):
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for x, y in test_data:
            output = net.forward(x.view(-1, 28*28))
            for i, out in enumerate(output):
                if out.argmax() == y[i]:
                    n_correct += 1
                n_total += 1
    return n_correct / n_total

def main():
    train_data = get_data_loader(is_train=True)
    test_data = get_data_loader(is_train=False)
    net = Net()

    print("Initial accuracy: ", evaluate(test_data, net) * 100)
    loss_function = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(2):
        for epoch in range(2):
            for x, y in train_data:
                output = net.forward(x.view(-1, 28*28))
                loss = loss_function(output, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch+1}, Loss: {loss.item()}")
            evaluate_model(net, test_data)
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
   
    accuracy = evaluate(test_data, net)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    
def evaluate_model(net, test_data):
    net.eval()
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for n, (x, y) in enumerate(test_data):
            output = net.forward(x.view(-1, 28*28))
            preds = output.argmax(dim=1)
            n_correct += (preds == y).sum().item()
            n_total += y.size(0)
    accuracy = n_correct / n_total if n_total > 0 else 0
    print(f"Evaluate Model Accuracy: {accuracy * 100:.2f}%")
    return accuracy
            

if __name__ == "__main__":
    main()