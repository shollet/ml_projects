import random
import numpy as np
import torch
from typing import Tuple, List, NamedTuple, Type
from tqdm import tqdm
import torchvision
from torchvision import transforms
from collections import OrderedDict

# Seed all random number generators
np.random.seed(197331)
torch.manual_seed(197331)
random.seed(197331)

class NetworkConfiguration(NamedTuple):
    n_channels: Tuple[int, ...] = (16, 32, 48)
    kernel_sizes: Tuple[int, ...] = (3, 3, 3)
    strides: Tuple[int, ...] = (1, 1, 1)
    dense_hiddens: Tuple[int, ...] = (256, 256)

class Trainer:
    def __init__(self,
                 network_type: str = "mlp",
                 net_config: NetworkConfiguration = NetworkConfiguration(),
                 lr: float = 0.001,
                 batch_size: int = 128,
                 activation_name: str = "relu"):

        self.lr = lr
        self.batch_size = batch_size
        self.train, self.test = self.load_dataset(self)
        dataiter = iter(self.train)
        images, labels = next(dataiter)
        input_dim = images.shape[1:]
        self.network_type = network_type
        activation_function = self.create_activation_function(activation_name)
        if network_type == "mlp":
            self.network = self.create_mlp(input_dim[0]*input_dim[1]*input_dim[2], 
                                           net_config,
                                           activation_function)
        elif network_type == "cnn":
            self.network = self.create_cnn(input_dim[0], 
                                           net_config, 
                                           activation_function)
        else:
            raise ValueError("Network type not supported")
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)

        self.train_logs = {'train_loss': [], 'test_loss': [],
                           'train_mae': [], 'test_mae': []}

    @staticmethod
    def load_dataset(self):
        transform = transforms.ToTensor()

        trainset = torchvision.datasets.SVHN(root='./data', split='train',
                                             download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size,
                                                  shuffle=True)

        testset = torchvision.datasets.SVHN(root='./data', split='test',
                                            download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size,
                                                 shuffle=False)

        return trainloader, testloader

    @staticmethod
    def create_mlp(input_dim: int, net_config: NetworkConfiguration, 
                   activation: Type[torch.nn.Module]) -> torch.nn.Module:
        if input_dim <= 0:
            raise ValueError("input_dim doit être un entier positif.")

        layers = []
        layers.append(('flatten', torch.nn.Flatten()))
        hidden_sizes = net_config.dense_hiddens

        if not all(h > 0 for h in hidden_sizes):
            raise ValueError("Toutes les valeurs de dense_hiddens doivent être des entiers positifs.")
        
        layer_sizes = [input_dim] + list(hidden_sizes) + [1]
        
        for i in range(len(layer_sizes) - 1):
            layers.append((f'linear{i+1}', torch.nn.Linear(layer_sizes[i], layer_sizes[i+1])))
            
            if i < len(layer_sizes) - 2:
                layers.append((f'activation{i+1}', activation()))
        
        return torch.nn.Sequential(OrderedDict(layers))

    @staticmethod
    def create_cnn(in_channels: int, net_config: NetworkConfiguration, 
                   activation: Type[torch.nn.Module]) -> torch.nn.Module:
        """
        Crée un CNN selon la configuration spécifiée.
        
        Args:
            in_channels: Nombre de canaux de l'image d'entrée
            net_config: Configuration du réseau (utilise n_channels, kernel_sizes, strides, dense_hiddens)
            activation: Classe de la fonction d'activation à utiliser entre les couches
            
        Returns:
            Un module PyTorch implémentant le CNN
        """
        layers = []
        current_channels = in_channels

        # 1. Construction des couches convolutives
        for i in range(len(net_config.n_channels)):
            # Couche convolutive
            conv = torch.nn.Conv2d(
                in_channels=current_channels,
                out_channels=net_config.n_channels[i],
                kernel_size=net_config.kernel_sizes[i],
                stride=net_config.strides[i],
                padding=net_config.kernel_sizes[i] // 2  # padding pour maintenir la dimension
            )
            layers.append(conv)
            layers.append(activation())  # Créer une nouvelle instance
            
            # MaxPool pour toutes les couches sauf la dernière
            if i < len(net_config.n_channels) - 1:
                layers.append(torch.nn.MaxPool2d(kernel_size=2))
            
            current_channels = net_config.n_channels[i]
        
        # 2. Après la dernière conv: AdaptiveMaxPool2d et Flatten
        layers.append(torch.nn.AdaptiveMaxPool2d((4, 4)))
        layers.append(torch.nn.Flatten())
        
        # 3. Calcul de la dimension d'entrée pour la partie dense
        dense_input_size = current_channels * 4 * 4  # après AdaptiveMaxPool2d(4,4)
        
        # 4. Construction des couches denses
        dense_sizes = [dense_input_size] + list(net_config.dense_hiddens) + [1]
        for i in range(len(dense_sizes) - 1):
            layers.append(torch.nn.Linear(dense_sizes[i], dense_sizes[i+1]))
            if i < len(dense_sizes) - 2:  # pas d'activation après la dernière couche
                layers.append(activation())  # Créer une nouvelle instance
        
        return torch.nn.Sequential(*layers)

    @staticmethod
    def create_activation_function(activation_str: str) -> Type[torch.nn.Module]:
        if not isinstance(activation_str, str):
            raise TypeError("Le paramètre 'activation_str' doit être une chaîne de caractères.")

        activation_str = activation_str.lower()
        activations = {
            "relu": torch.nn.ReLU,
            "tanh": torch.nn.Tanh,
            "sigmoid": torch.nn.Sigmoid
        }
        if activation_str in activations:
            return activations[activation_str]
        else:
            raise ValueError(f"Fonction d'activation '{activation_str}' non supportée. Utilisez 'relu', 'tanh' ou 'sigmoid'.")

    def compute_loss_and_mae(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO WRITE CODE HERE
        pass

    def training_step(self, X_batch: torch.Tensor, y_batch: torch.Tensor):
        # TODO WRITE CODE HERE
        pass

    def train_loop(self, n_epochs: int) -> dict:
        N = len(self.train)
        for epoch in tqdm(range(n_epochs)):
            train_loss = 0.0
            train_mae = 0.0
            for i, data in enumerate(self.train):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                loss, mae = self.training_step(inputs, labels)
                train_loss += loss
                train_mae += mae

            # Log data every epoch
            self.train_logs['train_mae'].append(train_mae / N)
            self.train_logs['train_loss'].append(train_loss / N)
            self.evaluation_loop()
    
        return self.train_logs

    def evaluation_loop(self) -> None:
        self.network.eval()
        N = len(self.test)
        with torch.inference_mode():
            test_loss = 0.0
            test_mae = 0.0
            for data in self.test:
                inputs, labels = data
                loss, mae = self.compute_loss_and_mae(inputs, labels)
                test_loss += loss.item()
                test_mae += mae.item()

        self.train_logs['test_mae'].append(test_mae / N)
        self.train_logs['test_loss'].append(test_loss / N)

    def evaluate(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO WRITE CODE HERE
        pass
