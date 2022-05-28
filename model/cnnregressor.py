import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from moduler import *
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model_effnet0 = models.efficientnet_b0(pretrained=True)

for p in model_effnet0.parameters():
  p.requires_grad = False

classifier_name, old_classifier = model_effnet0._modules.popitem()
classifier_input_size = old_classifier[1].in_features


classifier = nn.Sequential(OrderedDict([
                           ('0', nn.Dropout(p=0.2)),
                           ('1', nn.PReLU()),
                           ('2', nn.Linear(in_features=classifier_input_size, out_features=128, bias=True)),
                           ('3', nn.BatchNorm1d(128)),
                           ('4', nn.Linear(in_features=128, out_features=1, bias=True)),
                           ]))

model_effnet0.add_module(classifier_name, classifier)
model_effnet0 = model_effnet0.to(device)

print(f'Linear:\t{model_effnet0.classifier}')

class CNNRegressor(torch.nn.Module):
    def __init__(self):
        super(CNNRegressor, self).__init__()

        self.cnn = models.efficientnet_b0(pretrained=True)

        for p in self.cnn.parameters():
          p.requires_grad = False

        classifier_name, old_classifier = self.cnn._modules.popitem()
        classifier_input_size = old_classifier[1].in_features

        classifier = nn.Sequential(OrderedDict([
                                   ('0', nn.Dropout(p=0.2)),
                                   ('1', nn.PReLU()),
                                   ('2', nn.Linear(in_features=classifier_input_size, out_features=128, bias=True)),
                                   ('3', nn.BatchNorm1d(128)),
                                   ('4', nn.Linear(in_features=128, out_features=1, bias=True)),
                                   ]))
        
        self.cnn.add_module(classifier_name, classifier)

        self.cnn = self.cnn.to(device)
        self.fc1 = nn.Linear(1 + 1, 16)
        self.fc1 = self.fc1.to(device)

        self.fc2 = nn.BatchNorm1d(16)
        self.fc2 = self.fc2.to(device)

        self.fc3 = nn.Linear(16, 1)
        self.fc3 = self.fc3.to(device)

    def forward(self, image, data):
      x1 = self.cnn(image)
      x2 = data.to(device)

      x = torch.cat((x1, x2), dim=1)
      x = self.fc1(x)
      x = self.fc2(x)
      x = self.fc3(x)

      return x