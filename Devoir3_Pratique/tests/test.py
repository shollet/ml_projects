import unittest
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solution import Trainer, NetworkConfiguration

# 1. Test de la création de la fonction d'activation
class TestActivationFunction(unittest.TestCase):
    def test_relu(self):
        relu = Trainer.create_activation_function("ReLU")
        self.assertIs(relu, torch.nn.ReLU)

    def test_tanh(self):
        tanh = Trainer.create_activation_function("tanh")
        self.assertIs(tanh, torch.nn.Tanh)

    def test_sigmoid(self):
        sigmoid = Trainer.create_activation_function("SIGMOID")
        self.assertIs(sigmoid, torch.nn.Sigmoid)

    def test_invalid_activation(self):
        with self.assertRaises(ValueError):
            Trainer.create_activation_function("invalid")

# 2. Test de la création du modèle MLP
class TestMLPCreation(unittest.TestCase):
    def test_mlp_architecture(self):
        net_config = NetworkConfiguration(dense_hiddens=(256, 256))
        activation = Trainer.create_activation_function("ReLU")  # Now returns the class

        input_dim = 32 * 32 * 3
        model = Trainer.create_mlp(input_dim, net_config, activation)

        layers = list(model.children())
        print("Layers in the model:")
        for idx, layer in enumerate(layers):
            print(f"Layer {idx}: {layer}")

        # Continue avec les tests
        self.assertIsInstance(model, torch.nn.Sequential)
        self.assertIsInstance(layers[0], torch.nn.Flatten)
        self.assertIsInstance(layers[1], torch.nn.Linear)
        self.assertIsInstance(layers[2], torch.nn.ReLU)
        self.assertIsInstance(layers[3], torch.nn.Linear)
        self.assertIsInstance(layers[4], torch.nn.ReLU)
        self.assertIsInstance(layers[5], torch.nn.Linear)

        # Vérifier les dimensions des couches
        self.assertEqual(layers[1].in_features, input_dim)
        self.assertEqual(layers[1].out_features, 256)
        self.assertEqual(layers[3].in_features, 256)
        self.assertEqual(layers[3].out_features, 256)
        self.assertEqual(layers[5].in_features, 256)
        self.assertEqual(layers[5].out_features, 1)

        # Tester avec un batch d'entrée
        batch_size = 32
        x = torch.randn(batch_size, 3, 32, 32)
        output = model(x)
        self.assertEqual(output.shape, (batch_size, 1))

# 3. Test de la création du modèle CNN
class TestCNNCreation(unittest.TestCase):
    def test_cnn_architecture(self):
        # Configuration de test
        in_channels = 3
        net_config = NetworkConfiguration(
            n_channels=(16, 32),
            kernel_sizes=(3, 3),
            strides=(1, 1),
            dense_hiddens=(256,)
        )
        activation = torch.nn.ReLU  # Passer la classe

        # Créer le modèle
        model = Trainer.create_cnn(in_channels, net_config, activation)

        # 1. Vérifier que c'est un Sequential
        self.assertIsInstance(model, torch.nn.Sequential)

        # 2. Vérifier la structure
        layers = list(model.children())

        # Première couche convolutive
        self.assertIsInstance(layers[0], torch.nn.Conv2d)
        self.assertEqual(layers[0].in_channels, 3)
        self.assertEqual(layers[0].out_channels, 16)

        # Vérifier activation et MaxPool2d après première conv
        self.assertIsInstance(layers[1], torch.nn.ReLU)
        self.assertIsInstance(layers[2], torch.nn.MaxPool2d)

        # Deuxième couche convolutive
        self.assertIsInstance(layers[3], torch.nn.Conv2d)
        self.assertEqual(layers[3].in_channels, 16)
        self.assertEqual(layers[3].out_channels, 32)

        # Vérifier activation, AdaptiveMaxPool2d et Flatten après dernière conv
        self.assertIsInstance(layers[4], torch.nn.ReLU)
        self.assertIsInstance(layers[5], torch.nn.AdaptiveMaxPool2d)
        self.assertIsInstance(layers[6], torch.nn.Flatten)

        # Vérifier les couches denses
        self.assertIsInstance(layers[7], torch.nn.Linear)
        self.assertEqual(layers[7].in_features, 32 * 4 * 4)
        self.assertEqual(layers[7].out_features, 256)
        self.assertIsInstance(layers[8], torch.nn.ReLU)
        self.assertIsInstance(layers[9], torch.nn.Linear)
        self.assertEqual(layers[9].in_features, 256)
        self.assertEqual(layers[9].out_features, 1)

        # 3. Test avec différentes tailles d'entrée
        batch_size = 32
        for img_size in [(32, 32), (64, 64), (128, 128)]:
            x = torch.randn(batch_size, 3, *img_size)
            output = model(x)
            self.assertEqual(output.shape, (batch_size, 1))

if __name__ == '__main__':
    unittest.main()
