"""
AI-based analysis module for the Integrated Electrical-Thermal Impedance Analyzer

This module provides deep learning models for analyzing electrical and thermal
impedance data, including feature extraction, classification, and prediction
of system characteristics.

Based on the methodology described in the patent:
열 임피던스와 전기 임피던스 통합 분석 시스템 및 방법
(Integrated Electrical-Thermal Impedance Analysis System and Method)

Author: Jihwan Jang
Organization: Ucaretron Inc.
"""

import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
import pickle
import os
import time
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, mean_squared_error, r2_score,
                           confusion_matrix)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras import layers, Model, Input
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class ModelType(Enum):
    """Enumeration for different model types."""
    CNN = 1
    RNN = 2
    HYBRID = 3
    MLP = 4
    ENSEMBLE = 5


class TaskType(Enum):
    """Enumeration for different task types."""
    CLASSIFICATION = 1
    REGRESSION = 2
    ANOMALY_DETECTION = 3
    SYSTEM_IDENTIFICATION = 4


class Framework(Enum):
    """Enumeration for different deep learning frameworks."""
    PYTORCH = 1
    TENSORFLOW = 2
    SKLEARN = 3


class ImpedanceDataset:
    """Class for handling impedance datasets for deep learning models."""
    
    def __init__(self, X=None, y=None, test_size=0.2, random_state=42,
                normalize=True, scaler_type='standard'):
        """
        Initialize the dataset.
        
        Parameters
        ----------
        X : array_like, optional
            Input features (impedance data)
        y : array_like, optional
            Target values or labels
        test_size : float, optional
            Proportion of data to use for testing
        random_state : int, optional
            Random seed for reproducibility
        normalize : bool, optional
            Whether to normalize the data
        scaler_type : str, optional
            Type of scaler to use ('standard' or 'minmax')
        """
        self.X = X
        self.y = y
        self.test_size = test_size
        self.random_state = random_state
        self.normalize = normalize
        self.scaler_type = scaler_type
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        
        if X is not None and y is not None:
            self.prepare_data()
    
    def prepare_data(self):
        """Prepare the data for training and testing."""
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Normalize data if requested
        if self.normalize:
            if self.scaler_type == 'standard':
                self.scaler = StandardScaler()
            else:  # 'minmax'
                self.scaler = MinMaxScaler()
            
            # Reshape if needed
            orig_shape = self.X_train.shape
            if len(orig_shape) > 2:
                # Flatten for scaling
                X_train_flat = self.X_train.reshape(orig_shape[0], -1)
                X_test_flat = self.X_test.reshape(self.X_test.shape[0], -1)
                
                # Fit and transform
                X_train_scaled = self.scaler.fit_transform(X_train_flat)
                X_test_scaled = self.scaler.transform(X_test_flat)
                
                # Reshape back
                self.X_train = X_train_scaled.reshape(orig_shape)
                self.X_test = X_test_scaled.reshape(self.X_test.shape)
            else:
                # 2D data can be scaled directly
                self.X_train = self.scaler.fit_transform(self.X_train)
                self.X_test = self.scaler.transform(self.X_test)
    
    def load_data(self, X, y):
        """
        Load new data into the dataset.
        
        Parameters
        ----------
        X : array_like
            Input features (impedance data)
        y : array_like
            Target values or labels
        """
        self.X = X
        self.y = y
        self.prepare_data()
    
    def get_torch_dataloaders(self, batch_size=32, shuffle=True):
        """
        Convert the dataset to PyTorch DataLoaders.
        
        Parameters
        ----------
        batch_size : int, optional
            Batch size for training
        shuffle : bool, optional
            Whether to shuffle the training data
        
        Returns
        -------
        tuple
            Training and testing DataLoader objects
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available. Please install it first.")
        
        # Convert to torch tensors
        X_train_tensor = torch.tensor(self.X_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(self.X_test, dtype=torch.float32)
        
        # Handle different types of targets
        if isinstance(self.y_train[0], (int, np.integer)):
            # Classification with integer labels
            y_train_tensor = torch.tensor(self.y_train, dtype=torch.long)
            y_test_tensor = torch.tensor(self.y_test, dtype=torch.long)
        else:
            # Regression or multi-label classification
            y_train_tensor = torch.tensor(self.y_train, dtype=torch.float32)
            y_test_tensor = torch.tensor(self.y_test, dtype=torch.float32)
        
        # Create TensorDatasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        return train_loader, test_loader
    
    def get_tf_datasets(self, batch_size=32, shuffle=True):
        """
        Convert the dataset to TensorFlow Datasets.
        
        Parameters
        ----------
        batch_size : int, optional
            Batch size for training
        shuffle : bool, optional
            Whether to shuffle the training data
        
        Returns
        -------
        tuple
            Training and testing tf.data.Dataset objects
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is not available. Please install it first.")
        
        # Create TensorFlow datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((self.X_train, self.y_train))
        test_dataset = tf.data.Dataset.from_tensor_slices((self.X_test, self.y_test))
        
        # Configure datasets
        if shuffle:
            train_dataset = train_dataset.shuffle(buffer_size=len(self.X_train))
        
        train_dataset = train_dataset.batch(batch_size)
        test_dataset = test_dataset.batch(batch_size)
        
        return train_dataset, test_dataset
    
    def transform_new_data(self, X):
        """
        Transform new data using the fitted scaler.
        
        Parameters
        ----------
        X : array_like
            Input features to transform
        
        Returns
        -------
        array_like
            Transformed features
        """
        if self.scaler is None:
            raise ValueError("Scaler is not fitted. Please prepare data first.")
        
        # Check if reshaping is needed
        orig_shape = X.shape
        if len(orig_shape) > 2:
            # Flatten for scaling
            X_flat = X.reshape(orig_shape[0], -1)
            
            # Transform
            X_scaled = self.scaler.transform(X_flat)
            
            # Reshape back
            return X_scaled.reshape(orig_shape)
        else:
            # 2D data can be transformed directly
            return self.scaler.transform(X)
    
    def save(self, filepath):
        """
        Save the dataset to a file.
        
        Parameters
        ----------
        filepath : str
            Path to save the dataset
        """
        data = {
            'X_train': self.X_train,
            'X_test': self.X_test,
            'y_train': self.y_train,
            'y_test': self.y_test,
            'scaler': self.scaler,
            'normalize': self.normalize,
            'scaler_type': self.scaler_type,
            'test_size': self.test_size,
            'random_state': self.random_state
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, filepath):
        """
        Load a dataset from a file.
        
        Parameters
        ----------
        filepath : str
            Path to load the dataset from
        
        Returns
        -------
        ImpedanceDataset
            Loaded dataset
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        dataset = cls(normalize=data['normalize'], 
                    scaler_type=data['scaler_type'],
                    test_size=data['test_size'], 
                    random_state=data['random_state'])
        
        dataset.X_train = data['X_train']
        dataset.X_test = data['X_test']
        dataset.y_train = data['y_train']
        dataset.y_test = data['y_test']
        dataset.scaler = data['scaler']
        
        return dataset


class BaseModel:
    """Base class for all AI models."""
    
    def __init__(self, name='base_model', task_type=TaskType.CLASSIFICATION,
                framework=Framework.PYTORCH, save_dir='models'):
        """
        Initialize the base model.
        
        Parameters
        ----------
        name : str, optional
            Model name
        task_type : TaskType, optional
            Type of task (classification, regression, etc.)
        framework : Framework, optional
            Deep learning framework to use
        save_dir : str, optional
            Directory to save model checkpoints
        """
        self.name = name
        self.task_type = task_type
        self.framework = framework
        self.save_dir = save_dir
        self.model = None
        self.is_trained = False
        self.history = None
        
        # Create save directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    def build_model(self, input_shape, output_shape):
        """
        Build the model architecture.
        
        Parameters
        ----------
        input_shape : tuple
            Shape of input data
        output_shape : int or tuple
            Shape of output data
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def train(self, dataset, epochs=100, batch_size=32, verbose=1):
        """
        Train the model.
        
        Parameters
        ----------
        dataset : ImpedanceDataset
            Dataset to train on
        epochs : int, optional
            Number of training epochs
        batch_size : int, optional
            Batch size for training
        verbose : int, optional
            Verbosity level
        
        Returns
        -------
        dict
            Training history
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def predict(self, X):
        """
        Make predictions with the model.
        
        Parameters
        ----------
        X : array_like
            Input data
        
        Returns
        -------
        array_like
            Predictions
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def evaluate(self, X, y):
        """
        Evaluate model performance.
        
        Parameters
        ----------
        X : array_like
            Input data
        y : array_like
            Target values or labels
        
        Returns
        -------
        dict
            Evaluation metrics
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def save(self, filepath=None):
        """
        Save the model to a file.
        
        Parameters
        ----------
        filepath : str, optional
            Path to save the model. If None, use a default path based on model name.
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def load(self, filepath):
        """
        Load the model from a file.
        
        Parameters
        ----------
        filepath : str
            Path to load the model from
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_feature_importance(self):
        """
        Get feature importance if applicable.
        
        Returns
        -------
        array_like
            Feature importance scores
        """
        raise NotImplementedError("Feature importance not implemented for this model")


class PyTorchCNN(BaseModel):
    """Convolutional Neural Network implemented in PyTorch."""
    
    def __init__(self, name='cnn_model', task_type=TaskType.CLASSIFICATION,
                save_dir='models', num_classes=None, num_outputs=None):
        """
        Initialize the CNN model.
        
        Parameters
        ----------
        name : str, optional
            Model name
        task_type : TaskType, optional
            Type of task (classification, regression, etc.)
        save_dir : str, optional
            Directory to save model checkpoints
        num_classes : int, optional
            Number of classes for classification tasks
        num_outputs : int, optional
            Number of outputs for regression tasks
        """
        super().__init__(name, task_type, Framework.PYTORCH, save_dir)
        self.num_classes = num_classes
        self.num_outputs = num_outputs
        
        # Check if PyTorch is available
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available. Please install it first.")
    
    def build_model(self, input_shape, output_shape=None):
        """
        Build the CNN architecture.
        
        Parameters
        ----------
        input_shape : tuple
            Shape of input data (e.g., (channels, freq_points) or (channels, height, width))
        output_shape : int or tuple, optional
            Shape of output data. If None, use num_classes or num_outputs.
        
        Returns
        -------
        torch.nn.Module
            Built model
        """
        # Determine output size
        if output_shape is not None:
            if isinstance(output_shape, tuple):
                output_size = output_shape[0]
            else:
                output_size = output_shape
        elif self.num_classes is not None:
            output_size = self.num_classes
        elif self.num_outputs is not None:
            output_size = self.num_outputs
        else:
            raise ValueError("Either output_shape, num_classes, or num_outputs must be provided")
        
        # Determine input dimensions
        if len(input_shape) == 2:
            # 1D data (channels, features)
            channels, features = input_shape
            
            # Create 1D CNN
            class CNN1D(nn.Module):
                def __init__(self, in_channels, feature_length, output_size, task_type):
                    super(CNN1D, self).__init__()
                    
                    # Define layers
                    self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1)
                    self.bn1 = nn.BatchNorm1d(32)
                    self.pool1 = nn.MaxPool1d(2)
                    
                    self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
                    self.bn2 = nn.BatchNorm1d(64)
                    self.pool2 = nn.MaxPool1d(2)
                    
                    self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
                    self.bn3 = nn.BatchNorm1d(128)
                    self.pool3 = nn.MaxPool1d(2)
                    
                    # Calculate size after convolutions
                    feature_size = feature_length // 8  # After 3 pooling layers (2^3 = 8)
                    if feature_size < 1:
                        feature_size = 1
                    
                    # Fully connected layers
                    self.fc1 = nn.Linear(128 * feature_size, 128)
                    self.dropout = nn.Dropout(0.5)
                    self.fc2 = nn.Linear(128, output_size)
                    
                    self.task_type = task_type
                
                def forward(self, x):
                    # Convolutional layers
                    x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
                    x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
                    x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
                    
                    # Flatten
                    x = x.view(x.size(0), -1)
                    
                    # Fully connected layers
                    x = torch.relu(self.fc1(x))
                    x = self.dropout(x)
                    x = self.fc2(x)
                    
                    # Apply appropriate activation for task
                    if self.task_type == TaskType.CLASSIFICATION:
                        if output_size > 1:
                            x = torch.log_softmax(x, dim=1)
                    
                    return x
            
            self.model = CNN1D(channels, features, output_size, self.task_type)
        
        elif len(input_shape) == 3:
            # 2D data (channels, height, width)
            channels, height, width = input_shape
            
            # Create 2D CNN
            class CNN2D(nn.Module):
                def __init__(self, in_channels, height, width, output_size, task_type):
                    super(CNN2D, self).__init__()
                    
                    # Define layers
                    self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
                    self.bn1 = nn.BatchNorm2d(32)
                    self.pool1 = nn.MaxPool2d(2)
                    
                    self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
                    self.bn2 = nn.BatchNorm2d(64)
                    self.pool2 = nn.MaxPool2d(2)
                    
                    self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
                    self.bn3 = nn.BatchNorm2d(128)
                    self.pool3 = nn.MaxPool2d(2)
                    
                    # Calculate size after convolutions
                    h_out = height // 8  # After 3 pooling layers (2^3 = 8)
                    w_out = width // 8
                    
                    if h_out < 1: h_out = 1
                    if w_out < 1: w_out = 1
                    
                    # Fully connected layers
                    self.fc1 = nn.Linear(128 * h_out * w_out, 128)
                    self.dropout = nn.Dropout(0.5)
                    self.fc2 = nn.Linear(128, output_size)
                    
                    self.task_type = task_type
                
                def forward(self, x):
                    # Convolutional layers
                    x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
                    x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
                    x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
                    
                    # Flatten
                    x = x.view(x.size(0), -1)
                    
                    # Fully connected layers
                    x = torch.relu(self.fc1(x))
                    x = self.dropout(x)
                    x = self.fc2(x)
                    
                    # Apply appropriate activation for task
                    if self.task_type == TaskType.CLASSIFICATION:
                        if output_size > 1:
                            x = torch.log_softmax(x, dim=1)
                    
                    return x
            
            self.model = CNN2D(channels, height, width, output_size, self.task_type)
        
        else:
            raise ValueError(f"Unsupported input shape: {input_shape}")
        
        return self.model
    
    def train(self, dataset, epochs=100, batch_size=32, verbose=1, 
             learning_rate=0.001, weight_decay=0.0001):
        """
        Train the model.
        
        Parameters
        ----------
        dataset : ImpedanceDataset
            Dataset to train on
        epochs : int, optional
            Number of training epochs
        batch_size : int, optional
            Batch size for training
        verbose : int, optional
            Verbosity level
        learning_rate : float, optional
            Learning rate for the optimizer
        weight_decay : float, optional
            Weight decay (L2 penalty) for the optimizer
        
        Returns
        -------
        dict
            Training history
        """
        # Check if model is built
        if self.model is None:
            raise ValueError("Model is not built. Call build_model first.")
        
        # Get data loaders
        train_loader, test_loader = dataset.get_torch_dataloaders(batch_size)
        
        # Set up optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        if self.task_type == TaskType.CLASSIFICATION:
            if self.num_classes == 2 or self.num_outputs == 1:
                # Binary classification
                criterion = nn.BCEWithLogitsLoss()
            else:
                # Multi-class classification
                criterion = nn.NLLLoss()
        else:  # Regression
            criterion = nn.MSELoss()
        
        # Training loop
        self.model.train()
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_metrics': []
        }
        
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(epochs):
            train_loss = 0.0
            
            # Training step
            for inputs, targets in train_loader:
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Reshape targets if needed for the loss function
                if self.task_type == TaskType.CLASSIFICATION and self.num_classes == 2:
                    targets = targets.float().view(-1, 1)
                
                # Calculate loss
                loss = criterion(outputs, targets)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * inputs.size(0)
            
            train_loss /= len(train_loader.dataset)
            history['train_loss'].append(train_loss)
            
            # Validation step
            val_loss, val_metrics = self._validate(test_loader, criterion)
            history['val_loss'].append(val_loss)
            history['val_metrics'].append(val_metrics)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
            
            if verbose > 0 and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
                if self.task_type == TaskType.CLASSIFICATION:
                    print(f"  Validation accuracy: {val_metrics['accuracy']:.4f}")
                else:
                    print(f"  Validation R²: {val_metrics['r2']:.4f}")
        
        # Load best model
        self.model.load_state_dict(best_model_state)
        self.is_trained = True
        self.history = history
        
        return history
    
    def _validate(self, test_loader, criterion):
        """
        Validate the model on the test set.
        
        Parameters
        ----------
        test_loader : torch.utils.data.DataLoader
            Test data loader
        criterion : torch.nn.Module
            Loss function
        
        Returns
        -------
        tuple
            Validation loss and metrics dictionary
        """
        self.model.eval()
        val_loss = 0.0
        
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = self.model(inputs)
                
                # Reshape targets if needed for the loss function
                if self.task_type == TaskType.CLASSIFICATION and self.num_classes == 2:
                    loss_targets = targets.float().view(-1, 1)
                    loss = criterion(outputs, loss_targets)
                else:
                    loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                
                all_outputs.append(outputs)
                all_targets.append(targets)
        
        val_loss /= len(test_loader.dataset)
        
        # Concatenate batches
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Calculate metrics
        if self.task_type == TaskType.CLASSIFICATION:
            if self.num_classes == 2:
                # Binary classification
                predicted = (torch.sigmoid(all_outputs) > 0.5).long()
                predicted = predicted.view(-1)
            else:
                # Multi-class classification
                predicted = torch.exp(all_outputs).argmax(dim=1)
            
            accuracy = accuracy_score(all_targets.cpu().numpy(), predicted.cpu().numpy())
            precision = precision_score(all_targets.cpu().numpy(), predicted.cpu().numpy(), 
                                     average='weighted', zero_division=0)
            recall = recall_score(all_targets.cpu().numpy(), predicted.cpu().numpy(), 
                               average='weighted', zero_division=0)
            f1 = f1_score(all_targets.cpu().numpy(), predicted.cpu().numpy(), 
                        average='weighted', zero_division=0)
            
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        else:  # Regression
            mse = mean_squared_error(all_targets.cpu().numpy(), all_outputs.cpu().numpy())
            r2 = r2_score(all_targets.cpu().numpy(), all_outputs.cpu().numpy())
            
            metrics = {
                'mse': mse,
                'rmse': np.sqrt(mse),
                'r2': r2
            }
        
        self.model.train()
        return val_loss, metrics
    
    def predict(self, X):
        """
        Make predictions with the model.
        
        Parameters
        ----------
        X : array_like
            Input data
        
        Returns
        -------
        array_like
            Predictions
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet")
        
        # Convert input to tensor
        X_tensor = torch.tensor(X, dtype=torch.float32)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Make predictions
        with torch.no_grad():
            outputs = self.model(X_tensor)
            
            # Process outputs based on task type
            if self.task_type == TaskType.CLASSIFICATION:
                if self.num_classes == 2:
                    # Binary classification
                    predictions = (torch.sigmoid(outputs) > 0.5).long()
                else:
                    # Multi-class classification
                    predictions = torch.exp(outputs).argmax(dim=1)
            else:  # Regression
                predictions = outputs
        
        return predictions.cpu().numpy()
    
    def evaluate(self, X, y):
        """
        Evaluate model performance.
        
        Parameters
        ----------
        X : array_like
            Input data
        y : array_like
            Target values or labels
        
        Returns
        -------
        dict
            Evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet")
        
        # Convert input to tensor
        X_tensor = torch.tensor(X, dtype=torch.float32)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Make predictions
        with torch.no_grad():
            outputs = self.model(X_tensor)
        
        # Calculate metrics based on task type
        if self.task_type == TaskType.CLASSIFICATION:
            if self.num_classes == 2:
                # Binary classification
                probabilities = torch.sigmoid(outputs).cpu().numpy()
                predicted = (probabilities > 0.5).astype(int)
            else:
                # Multi-class classification
                probabilities = torch.exp(outputs).cpu().numpy()
                predicted = np.argmax(probabilities, axis=1)
            
            # Calculate classification metrics
            accuracy = accuracy_score(y, predicted)
            precision = precision_score(y, predicted, average='weighted', zero_division=0)
            recall = recall_score(y, predicted, average='weighted', zero_division=0)
            f1 = f1_score(y, predicted, average='weighted', zero_division=0)
            conf_matrix = confusion_matrix(y, predicted)
            
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'confusion_matrix': conf_matrix
            }
        else:  # Regression
            predictions = outputs.cpu().numpy()
            
            # Calculate regression metrics
            mse = mean_squared_error(y, predictions)
            r2 = r2_score(y, predictions)
            
            metrics = {
                'mse': mse,
                'rmse': np.sqrt(mse),
                'r2': r2
            }
        
        return metrics
    
    def save(self, filepath=None):
        """
        Save the model to a file.
        
        Parameters
        ----------
        filepath : str, optional
            Path to save the model. If None, use a default path based on model name.
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet")
        
        if filepath is None:
            filepath = os.path.join(self.save_dir, f"{self.name}.pt")
        
        # Save model state
        model_data = {
            'model_state': self.model.state_dict(),
            'task_type': self.task_type,
            'num_classes': self.num_classes,
            'num_outputs': self.num_outputs,
            'name': self.name,
            'history': self.history
        }
        
        torch.save(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """
        Load the model from a file.
        
        Parameters
        ----------
        filepath : str
            Path to load the model from
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # Load model data
        model_data = torch.load(filepath)
        
        # Update model attributes
        self.task_type = model_data['task_type']
        self.num_classes = model_data['num_classes']
        self.num_outputs = model_data['num_outputs']
        self.name = model_data['name']
        self.history = model_data['history']
        
        # Check if model is built
        if self.model is None:
            raise ValueError("Model architecture is not built. Call build_model first.")
        
        # Load model state
        self.model.load_state_dict(model_data['model_state'])
        self.is_trained = True
        
        print(f"Model loaded from {filepath}")


class ImpedanceNeuralNet:
    """
    Deep Neural Network for analyzing impedance data.
    
    This class provides high-level interface for building, training,
    and deploying deep learning models for impedance data analysis.
    It supports different model types and frameworks.
    """
    
    def __init__(self, model_type=ModelType.CNN, task_type=TaskType.CLASSIFICATION,
                framework=Framework.PYTORCH, name='impedance_model',
                num_classes=None, num_outputs=None, save_dir='models'):
        """
        Initialize the impedance neural network.
        
        Parameters
        ----------
        model_type : ModelType, optional
            Type of model to use
        task_type : TaskType, optional
            Type of task (classification, regression, etc.)
        framework : Framework, optional
            Deep learning framework to use
        name : str, optional
            Model name
        num_classes : int, optional
            Number of classes for classification tasks
        num_outputs : int, optional
            Number of outputs for regression tasks
        save_dir : str, optional
            Directory to save model checkpoints
        """
        self.model_type = model_type
        self.task_type = task_type
        self.framework = framework
        self.name = name
        self.num_classes = num_classes
        self.num_outputs = num_outputs
        self.save_dir = save_dir
        
        # Initialize model
        self._init_model()
    
    def _init_model(self):
        """Initialize the appropriate model based on type and framework."""
        if self.framework == Framework.PYTORCH:
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch is not available. Please install it first.")
            
            if self.model_type == ModelType.CNN:
                self.model = PyTorchCNN(name=self.name, task_type=self.task_type,
                                      save_dir=self.save_dir, num_classes=self.num_classes,
                                      num_outputs=self.num_outputs)
            elif self.model_type == ModelType.RNN:
                # Implement PyTorch RNN model
                raise NotImplementedError("PyTorch RNN model not implemented yet")
            elif self.model_type == ModelType.HYBRID:
                # Implement PyTorch hybrid model
                raise NotImplementedError("PyTorch hybrid model not implemented yet")
            elif self.model_type == ModelType.MLP:
                # Implement PyTorch MLP model
                raise NotImplementedError("PyTorch MLP model not implemented yet")
            elif self.model_type == ModelType.ENSEMBLE:
                # Implement PyTorch ensemble model
                raise NotImplementedError("PyTorch ensemble model not implemented yet")
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
        
        elif self.framework == Framework.TENSORFLOW:
            if not TF_AVAILABLE:
                raise ImportError("TensorFlow is not available. Please install it first.")
            
            # Implement TensorFlow models
            raise NotImplementedError("TensorFlow models not implemented yet")
        
        elif self.framework == Framework.SKLEARN:
            # Implement scikit-learn models
            raise NotImplementedError("scikit-learn models not implemented yet")
        
        else:
            raise ValueError(f"Unsupported framework: {self.framework}")
    
    def build(self, input_shape, output_shape=None):
        """
        Build the model architecture.
        
        Parameters
        ----------
        input_shape : tuple
            Shape of input data (without batch dimension)
        output_shape : int or tuple, optional
            Shape of output data. If None, use num_classes or num_outputs.
        
        Returns
        -------
        model
            Built model
        """
        return self.model.build_model(input_shape, output_shape)
    
    def train(self, dataset, epochs=100, batch_size=32, verbose=1, **kwargs):
        """
        Train the model.
        
        Parameters
        ----------
        dataset : ImpedanceDataset
            Dataset to train on
        epochs : int, optional
            Number of training epochs
        batch_size : int, optional
            Batch size for training
        verbose : int, optional
            Verbosity level
        **kwargs
            Additional parameters for the specific model's training method
        
        Returns
        -------
        dict
            Training history
        """
        return self.model.train(dataset, epochs, batch_size, verbose, **kwargs)
    
    def predict(self, X):
        """
        Make predictions with the model.
        
        Parameters
        ----------
        X : array_like
            Input data
        
        Returns
        -------
        array_like
            Predictions
        """
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        """
        Evaluate model performance.
        
        Parameters
        ----------
        X : array_like
            Input data
        y : array_like
            Target values or labels
        
        Returns
        -------
        dict
            Evaluation metrics
        """
        return self.model.evaluate(X, y)
    
    def save(self, filepath=None):
        """
        Save the model to a file.
        
        Parameters
        ----------
        filepath : str, optional
            Path to save the model. If None, use a default path based on model name.
        """
        self.model.save(filepath)
    
    def load(self, filepath):
        """
        Load the model from a file.
        
        Parameters
        ----------
        filepath : str
            Path to load the model from
        """
        self.model.load(filepath)


# Utility functions for analyzing impedance data with AI
def prepare_impedance_features(elec_freq, elec_imp, thermal_freq=None, thermal_imp=None,
                             feature_type='combined', num_features=100):
    """
    Prepare features from impedance data for AI models.
    
    Parameters
    ----------
    elec_freq : array_like
        Electrical frequency array
    elec_imp : array_like
        Electrical impedance array
    thermal_freq : array_like, optional
        Thermal frequency array
    thermal_imp : array_like, optional
        Thermal impedance array
    feature_type : str, optional
        Type of features to extract ('electrical', 'thermal', or 'combined')
    num_features : int, optional
        Number of features to generate
    
    Returns
    -------
    array_like
        Extracted features
    """
    # Check if we have the required data
    if feature_type in ['thermal', 'combined'] and (thermal_freq is None or thermal_imp is None):
        raise ValueError("Thermal impedance data required but not provided")
    
    if feature_type == 'electrical':
        # Extract features from electrical impedance only
        features = extract_electrical_features(elec_freq, elec_imp, num_features)
    
    elif feature_type == 'thermal':
        # Extract features from thermal impedance only
        features = extract_thermal_features(thermal_freq, thermal_imp, num_features)
    
    elif feature_type == 'combined':
        # Extract and combine features from both electrical and thermal impedance
        elec_features = extract_electrical_features(elec_freq, elec_imp, num_features // 2)
        thermal_features = extract_thermal_features(thermal_freq, thermal_imp, num_features // 2)
        
        # Combine features
        features = np.concatenate([elec_features, thermal_features], axis=1)
    
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")
    
    return features


def extract_electrical_features(freq, imp, num_features=50):
    """
    Extract features from electrical impedance data.
    
    Parameters
    ----------
    freq : array_like
        Frequency array
    imp : array_like
        Impedance array
    num_features : int, optional
        Number of features to extract
    
    Returns
    -------
    array_like
        Extracted features
    """
    # Extract magnitude and phase
    mag = np.abs(imp)
    phase = np.angle(imp, deg=True)
    
    # Ensure 2D shape for single sample
    if len(mag.shape) == 1:
        mag = mag.reshape(1, -1)
        phase = phase.reshape(1, -1)
    
    # Determine number of samples
    num_samples = mag.shape[0]
    
    # Interpolate to fixed number of points if needed
    if mag.shape[1] != num_features // 2:
        # Create log-spaced frequency grid
        log_freq = np.logspace(np.log10(freq.min()), np.log10(freq.max()), num_features // 2)
        
        # Interpolate each sample
        mag_interp = np.zeros((num_samples, num_features // 2))
        phase_interp = np.zeros((num_samples, num_features // 2))
        
        for i in range(num_samples):
            mag_interp[i] = np.interp(log_freq, freq, mag[i])
            phase_interp[i] = np.interp(log_freq, freq, phase[i])
        
        mag = mag_interp
        phase = phase_interp
    
    # Combine magnitude and phase features
    features = np.concatenate([mag, phase], axis=1)
    
    return features


def extract_thermal_features(freq, imp, num_features=50):
    """
    Extract features from thermal impedance data.
    
    Parameters
    ----------
    freq : array_like
        Frequency array
    imp : array_like
        Impedance array
    num_features : int, optional
        Number of features to extract
    
    Returns
    -------
    array_like
        Extracted features
    """
    # Similar approach as for electrical features, but may use different 
    # transformations specific to thermal data
    
    # Extract magnitude and phase
    mag = np.abs(imp)
    phase = np.angle(imp, deg=True)
    
    # Ensure 2D shape for single sample
    if len(mag.shape) == 1:
        mag = mag.reshape(1, -1)
        phase = phase.reshape(1, -1)
    
    # Determine number of samples
    num_samples = mag.shape[0]
    
    # Interpolate to fixed number of points if needed
    if mag.shape[1] != num_features // 2:
        # Create log-spaced frequency grid
        log_freq = np.logspace(np.log10(freq.min()), np.log10(freq.max()), num_features // 2)
        
        # Interpolate each sample
        mag_interp = np.zeros((num_samples, num_features // 2))
        phase_interp = np.zeros((num_samples, num_features // 2))
        
        for i in range(num_samples):
            mag_interp[i] = np.interp(log_freq, freq, mag[i])
            phase_interp[i] = np.interp(log_freq, freq, phase[i])
        
        mag = mag_interp
        phase = phase_interp
    
    # Combine magnitude and phase features
    features = np.concatenate([mag, phase], axis=1)
    
    return features


def batch_normalize_impedance(impedance_data, method='standard'):
    """
    Normalize batch of impedance data.
    
    Parameters
    ----------
    impedance_data : array_like
        Batch of impedance data
    method : str, optional
        Normalization method ('standard', 'minmax', 'robust')
    
    Returns
    -------
    array_like
        Normalized impedance data
    """
    # Check dimensionality
    if len(impedance_data.shape) < 2:
        raise ValueError("Expected at least 2D array")
    
    # Initialize scaler
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    # Reshape if needed
    orig_shape = impedance_data.shape
    if len(orig_shape) > 2:
        # Flatten all but first dimension
        flat_data = impedance_data.reshape(orig_shape[0], -1)
        # Normalize
        norm_data = scaler.fit_transform(flat_data)
        # Reshape back
        norm_data = norm_data.reshape(orig_shape)
    else:
        # Standard 2D case
        norm_data = scaler.fit_transform(impedance_data)
    
    return norm_data


def visualize_impedance_clusters(features, labels, method='tsne', title=None,
                               cmap='viridis', figsize=(10, 8)):
    """
    Visualize clusters of impedance data using dimensionality reduction.
    
    Parameters
    ----------
    features : array_like
        Feature matrix (samples x features)
    labels : array_like
        Cluster labels or class labels
    method : str, optional
        Dimensionality reduction method ('tsne', 'pca', or 'umap')
    title : str, optional
        Plot title
    cmap : str, optional
        Colormap for plotting
    figsize : tuple, optional
        Figure size
    
    Returns
    -------
    tuple
        Figure and axes objects
    """
    # Check if we have the required libraries
    if method == 'tsne':
        try:
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42)
        except ImportError:
            raise ImportError("scikit-learn is required for t-SNE visualization")
    
    elif method == 'pca':
        try:
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2)
        except ImportError:
            raise ImportError("scikit-learn is required for PCA visualization")
    
    elif method == 'umap':
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42)
        except ImportError:
            raise ImportError("umap-learn is required for UMAP visualization")
    
    else:
        raise ValueError(f"Unknown dimensionality reduction method: {method}")
    
    # Reduce dimensionality
    reduced_features = reducer.fit_transform(features)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    scatter = ax.scatter(reduced_features[:, 0], reduced_features[:, 1], 
                       c=labels, cmap=cmap, alpha=0.8, edgecolors='w', linewidth=0.5)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Class/Cluster')
    
    # Set labels and title
    ax.set_xlabel(f"{method.upper()} Dimension 1")
    ax.set_ylabel(f"{method.upper()} Dimension 2")
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Impedance Data Visualization using {method.upper()}")
    
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    
    return fig, ax


def plot_confusion_matrix(y_true, y_pred, class_names=None, figsize=(10, 8), 
                         title='Confusion Matrix', normalize=True, cmap='Blues'):
    """
    Plot confusion matrix for classification results.
    
    Parameters
    ----------
    y_true : array_like
        True labels
    y_pred : array_like
        Predicted labels
    class_names : list, optional
        Names for the classes
    figsize : tuple, optional
        Figure size
    title : str, optional
        Plot title
    normalize : bool, optional
        Whether to normalize the confusion matrix
    cmap : str, optional
        Colormap for plotting
    
    Returns
    -------
    tuple
        Figure and axes objects
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    if normalize:
        cbar.set_label('Normalized Counts')
    else:
        cbar.set_label('Counts')
    
    # Set up axes
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]
    
    ax.set_xticks(np.arange(cm.shape[1]))
    ax.set_yticks(np.arange(cm.shape[0]))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                  ha="center", va="center",
                  color="white" if cm[i, j] > thresh else "black")
    
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(title)
    fig.tight_layout()
    
    return fig, ax


def explain_model_predictions(model, X, feature_names=None, num_features=10):
    """
    Explain model predictions using feature importance or other techniques.
    
    Parameters
    ----------
    model : BaseModel or sklearn model
        Trained model to explain
    X : array_like
        Input data to explain predictions for
    feature_names : list, optional
        Names of the features
    num_features : int, optional
        Number of top features to show
    
    Returns
    -------
    dict
        Explanation results
    """
    # Try different explanation methods based on model type
    try:
        # Try to use model's built-in feature importance
        if hasattr(model, 'get_feature_importance'):
            importances = model.get_feature_importance()
            explanation_method = 'feature_importance'
        
        # For scikit-learn models
        elif hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            explanation_method = 'feature_importances_'
        
        # For linear models
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)
            explanation_method = 'coefficients'
        
        # Try SHAP if available
        else:
            try:
                import shap
                
                # Create explainer
                explainer = shap.Explainer(model)
                
                # Calculate SHAP values
                shap_values = explainer(X)
                
                # Get mean absolute SHAP values as feature importance
                importances = np.abs(shap_values.values).mean(axis=0)
                explanation_method = 'shap'
            
            except ImportError:
                return {"error": "No explanation method found and SHAP is not available"}
    
    except Exception as e:
        return {"error": f"Error calculating feature importance: {str(e)}"}
    
    # Create feature names if not provided
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(importances))]
    
    # Ensure feature_names and importances have the same length
    if len(feature_names) != len(importances):
        feature_names = feature_names[:len(importances)] if len(feature_names) > len(importances) else feature_names + [f"Feature {i}" for i in range(len(feature_names), len(importances))]
    
    # Get indices of top features
    indices = np.argsort(importances)[-num_features:]
    
    # Create result dictionary
    result = {
        "method": explanation_method,
        "importances": importances,
        "top_features": {feature_names[i]: importances[i] for i in indices[::-1]},
        "all_features": {feature_names[i]: importances[i] for i in range(len(importances))}
    }
    
    return result"""