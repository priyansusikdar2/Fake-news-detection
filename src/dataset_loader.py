"""
Advanced dataset loader with cross-validation and augmentation
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import StratifiedKFold, train_test_split
from transformers import AutoTokenizer
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FakeNewsDataset(Dataset):
    """PyTorch Dataset for fake news detection"""
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: AutoTokenizer,
        max_length: int = 256,
        augment: bool = False
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment
        
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Apply augmentation if enabled
        if self.augment:
            text = self._augment_text(text)
        
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
    
    def _augment_text(self, text: str) -> str:
        """Simple text augmentation"""
        # Implement synonym replacement, random deletion, etc.
        import random
        words = text.split()
        
        # Random word deletion (10% chance)
        if len(words) > 10 and random.random() < 0.1:
            delete_idx = random.randint(0, len(words) - 1)
            words.pop(delete_idx)
        
        return ' '.join(words)


class DataLoaderBuilder:
    """Build data loaders with cross-validation support"""
    
    def __init__(
        self,
        data_path: str = '../dataset/',
        model_name: str = 'bert-base-uncased',
        max_length: int = 256,
        batch_size: int = 32,
        num_workers: int = 2
    ):
        self.data_path = Path(data_path)
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load raw CSV files"""
        try:
            fake_df = pd.read_csv(self.data_path / 'Fake.csv')
            true_df = pd.read_csv(self.data_path / 'True.csv')
            logger.info(f"Loaded {len(fake_df)} fake and {len(true_df)} real articles")
            return fake_df, true_df
        except FileNotFoundError:
            logger.warning("Dataset not found, creating sample data")
            return self._create_sample_data()
    
    def _create_sample_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create sample data for testing"""
        np.random.seed(42)
        n_samples = 1000
        
        fake_texts = [f"Fake news sample {i}. " + " ".join(["fake", "false", "misleading"] * 10) 
                     for i in range(n_samples)]
        true_texts = [f"Real news sample {i}. " + " ".join(["truth", "fact", "verified"] * 10) 
                     for i in range(n_samples)]
        
        fake_df = pd.DataFrame({'text': fake_texts, 'label': 0})
        true_df = pd.DataFrame({'text': true_texts, 'label': 1})
        
        return fake_df, true_df
    
    def prepare_data(
        self,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42
    ) -> Dict[str, Dataset]:
        """Prepare train/validation/test datasets"""
        # Load and combine data
        fake_df, true_df = self.load_raw_data()
        df = pd.concat([fake_df, true_df], ignore_index=True)
        
        texts = df['text'].values
        labels = df['label'].values
        
        # First split: train+val vs test
        train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
            texts, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
        
        # Second split: train vs validation
        val_relative_size = val_size / (1 - test_size)
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_val_texts, train_val_labels, test_size=val_relative_size, 
            random_state=random_state, stratify=train_val_labels
        )
        
        logger.info(f"Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")
        
        # Create datasets
        train_dataset = FakeNewsDataset(
            train_texts, train_labels, self.tokenizer, self.max_length, augment=True
        )
        val_dataset = FakeNewsDataset(
            val_texts, val_labels, self.tokenizer, self.max_length, augment=False
        )
        test_dataset = FakeNewsDataset(
            test_texts, test_labels, self.tokenizer, self.max_length, augment=False
        )
        
        return {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        }
    
    def create_loaders(
        self,
        datasets: Dict[str, Dataset],
        batch_size: Optional[int] = None
    ) -> Dict[str, DataLoader]:
        """Create DataLoaders from datasets"""
        batch_size = batch_size or self.batch_size
        
        loaders = {}
        for name, dataset in datasets.items():
            shuffle = (name == 'train')
            loaders[name] = DataLoader(
                dataset,
                batch_size=batch_size if name != 'test' else batch_size * 2,
                shuffle=shuffle,
                num_workers=self.num_workers,
                pin_memory=True
            )
        
        return loaders
    
    def get_kfold_loaders(
        self,
        n_folds: int = 5,
        batch_size: Optional[int] = None
    ) -> List[Dict[str, DataLoader]]:
        """Create K-Fold cross-validation loaders"""
        # Load combined data
        fake_df, true_df = self.load_raw_data()
        df = pd.concat([fake_df, true_df], ignore_index=True)
        
        texts = df['text'].values
        labels = df['label'].values
        
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        fold_loaders = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
            train_texts, val_texts = texts[train_idx], texts[val_idx]
            train_labels, val_labels = labels[train_idx], labels[val_idx]
            
            train_dataset = FakeNewsDataset(
                train_texts, train_labels, self.tokenizer, self.max_length, augment=True
            )
            val_dataset = FakeNewsDataset(
                val_texts, val_labels, self.tokenizer, self.max_length, augment=False
            )
            
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size or self.batch_size, 
                shuffle=True, num_workers=self.num_workers
            )
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size or self.batch_size * 2, 
                shuffle=False, num_workers=self.num_workers
            )
            
            fold_loaders.append({'train': train_loader, 'val': val_loader, 'fold': fold})
        
        return fold_loaders