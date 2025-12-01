from .cnn_1d import CNN1D
from .resnet_1d import ResNet1D
from .lstm_model import BiLSTMModel
from .cnn_transformer import CNNTransformer

MODEL_REGISTRY = {
    "cnn": CNN1D,
    "resnet": ResNet1D,
    "lstm": BiLSTMModel,
    "transformer": CNNTransformer,
}
