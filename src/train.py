"""
Complete training script for fake news detection
Run with: python train.py
"""

import os
import sys
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FakeNewsDataset(Dataset):
    """PyTorch Dataset for fake news detection"""
    
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class ModelTrainer:
    """Complete training pipeline"""
    
    def __init__(
        self,
        data_path: str = '../dataset/',
        model_name: str = 'distilbert-base-uncased',  # Smaller model for faster training
        num_classes: int = 2,
        max_length: int = 128,
        batch_size: int = 16,
        epochs: int = 3,
        learning_rate: float = 2e-5,
        validation_split: float = 0.2,
        model_save_path: str = '../model/'
    ):
        self.data_path = Path(data_path)
        self.model_name = model_name
        self.num_classes = num_classes
        self.max_length = max_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.validation_split = validation_split
        self.model_save_path = Path(model_save_path)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Create save directory
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize tokenizer and model
        self.tokenizer = None
        self.model = None
        self.train_loader = None
        self.val_loader = None
        
    def load_data(self):
        """Load and prepare the dataset"""
        logger.info("Loading data...")
        
        # Try to load real data
        try:
            fake_df = pd.read_csv(self.data_path / 'Fake.csv')
            true_df = pd.read_csv(self.data_path / 'True.csv')
            logger.info(f"Loaded {len(fake_df)} fake and {len(true_df)} real articles")
            
            # Add labels
            fake_df['label'] = 0
            true_df['label'] = 1
            
            # Combine
            df = pd.concat([fake_df, true_df], ignore_index=True)
            
        except FileNotFoundError:
            logger.warning("Dataset not found. Creating sample data for testing...")
            # Create sample data
            np.random.seed(42)
            n_samples = 1000
            
            fake_texts = [f"Fake news sample {i}. " + " ".join(["fake", "lie", "false"] * 10) for i in range(n_samples)]
            true_texts = [f"Real news sample {i}. " + " ".join(["truth", "fact", "verified"] * 10) for i in range(n_samples)]
            
            fake_df = pd.DataFrame({'text': fake_texts, 'label': 0})
            true_df = pd.DataFrame({'text': true_texts, 'label': 1})
            df = pd.concat([fake_df, true_df], ignore_index=True)
        
        # Shuffle the data
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Split into train and validation
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            df['text'].values,
            df['label'].values,
            test_size=self.validation_split,
            random_state=42,
            stratify=df['label'].values
        )
        
        logger.info(f"Training samples: {len(train_texts)}")
        logger.info(f"Validation samples: {len(val_texts)}")
        logger.info(f"Fake: {sum(df['label']==0)}, Real: {sum(df['label']==1)}")
        
        return train_texts, train_labels, val_texts, val_labels
    
    def create_dataloaders(self, train_texts, train_labels, val_texts, val_labels):
        """Create DataLoaders"""
        logger.info("Creating dataloaders...")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Create datasets
        train_dataset = FakeNewsDataset(train_texts, train_labels, self.tokenizer, self.max_length)
        val_dataset = FakeNewsDataset(val_texts, val_labels, self.tokenizer, self.max_length)
        
        # Create dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0  # Set to 0 to avoid multiprocessing issues on Windows
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=0
        )
        
        logger.info(f"Train batches: {len(self.train_loader)}")
        logger.info(f"Validation batches: {len(self.val_loader)}")
    
    def initialize_model(self):
        """Initialize the model"""
        logger.info(f"Loading model: {self.model_name}")
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_classes
        )
        self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        # Initialize optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )
        
        # Initialize scheduler
        total_steps = len(self.train_loader) * self.epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        progress_bar = tqdm(self.train_loader, desc='Training')
        
        for batch in progress_bar:
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            # Track metrics
            total_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                total_loss += loss.item()
                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='binary')
        
        return avg_loss, accuracy, f1, all_preds, all_labels
    
    def train(self):
        """Complete training pipeline"""
        # Load data
        train_texts, train_labels, val_texts, val_labels = self.load_data()
        
        # Create dataloaders
        self.create_dataloaders(train_texts, train_labels, val_texts, val_labels)
        
        # Initialize model
        self.initialize_model()
        
        # Training history
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': []
        }
        
        best_accuracy = 0
        best_f1 = 0
        
        logger.info("\n" + "="*60)
        logger.info("STARTING TRAINING")
        logger.info("="*60)
        
        for epoch in range(self.epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.epochs}")
            logger.info("-" * 40)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            history['train_loss'].append(train_loss)
            history['train_accuracy'].append(train_acc)
            
            # Validate
            val_loss, val_acc, val_f1, _, _ = self.validate()
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)
            history['val_f1'].append(val_f1)
            
            logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
            
            # Save best model
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                best_f1 = val_f1
                self.save_model('best_model.pt')
                logger.info(f"✓ Saved best model (accuracy: {best_accuracy:.4f}, f1: {best_f1:.4f})")
        
        # Save final model
        self.save_model('final_model.pt')
        
        # Save training history
        self.save_history(history)
        
        # Final evaluation
        logger.info("\n" + "="*60)
        logger.info("FINAL EVALUATION ON VALIDATION SET")
        logger.info("="*60)
        
        _, _, _, all_preds, all_labels = self.validate()
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=['Fake', 'Real']))
        
        return history, best_accuracy, best_f1
    
    def save_model(self, filename):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'tokenizer': self.tokenizer,
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'max_length': self.max_length
        }
        torch.save(checkpoint, self.model_save_path / filename)
        logger.info(f"Model saved to {self.model_save_path / filename}")
    
    def save_history(self, history):
        """Save training history"""
        with open(self.model_save_path / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        logger.info(f"Training history saved to {self.model_save_path / 'training_history.json'}")


def main():
    """Main training function"""
    print("="*60)
    print("FAKE NEWS DETECTION - TRAINING SCRIPT")
    print("="*60)
    
    # Create trainer
    trainer = ModelTrainer(
        data_path='../dataset/',  # Adjust if your dataset is in a different location
        model_name='distilbert-base-uncased',  # Use smaller model for faster training
        batch_size=16,
        epochs=3,
        max_length=128
    )
    
    # Train model
    history, best_acc, best_f1 = trainer.train()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Best Validation Accuracy: {best_acc:.4f}")
    print(f"Best Validation F1-Score: {best_f1:.4f}")
    print(f"Model saved to: {trainer.model_save_path}")
    
    return trainer


if __name__ == "__main__":
    trainer = main()