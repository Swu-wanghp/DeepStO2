import torch
import torch.nn as nn
import torchvision

# 存储模型名称以及对应的权重名称
ModelWeights = {
    "mobilenet_v2": "MobileNet_V2_Weights"

}


class StO2Model(nn.Module):

    def __init__(self, model_name, num_classes, is_pretrained=False):
        super(StO2Model, self).__init__()
        self.model_name = model_name
        self.num_class = num_classes
        self.is_pretrained = is_pretrained

        print("ModelWeights.keys() = ", ModelWeights.keys())
        print("ModelWeights = ", ModelWeights)
        if self.model_name not in ModelWeights.keys():
            raise ValueError("不存在该名称对应的模型，请检查！")

        # getattr 获取对象的属性值
        if is_pretrained:
            self.base_model = getattr(torchvision.models, self.model_name)(weights=ModelWeights[self.model_name])
        else:
            self.base_model = getattr(torchvision.models, self.model_name)()

        if hasattr(self.base_model, 'classifier'):
            self.base_model.last_layer_name = 'classifier'
            feature_dim = self.base_model.classifier[1].in_features
            self.base_model.classifier = nn.Linear(feature_dim, self.num_class)
        elif hasattr(self.base_model, 'fc'):
            self.base_model.last_layer_name = 'fc'
            feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
            self.base_model.fc = nn.Linear(feature_dim, self.num_class)
        elif hasattr(self.base_model, 'head'):
            self.base_model.last_layer_name = 'head'
            feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
            self.base_model.head = nn.Linear(feature_dim, self.num_class)
        elif hasattr(self.base_model, 'heads'):
            self.base_model.last_layer_name = 'heads'
            feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
            self.base_model.heads = nn.Linear(feature_dim, self.num_class)
        else:
            raise ValueError('Please confirm the name of last')

    def forward(self, x):
        x = self.base_model(x)
        return x


if __name__ == "__main__":
    model_name = "mobilenet_v2"
    num_classes = 2
    is_pretrained = False

    clsmodel = StO2Model(model_name, num_classes, is_pretrained)
    print(clsmodel)
