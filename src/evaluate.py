"""
Evaluation script for trained fake news detection model
Run with: python evaluate.py
"""

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def load_model_and_data(model_path='../model/best_model.pt', data_path='../dataset/'):
    """Load trained model and test data"""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    
    print("Loading model...")
    
    # Load model
    if Path(model_path).exists():
        checkpoint = torch.load(model_path, map_location='cpu')
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint.get('model_name', 'distilbert-base-uncased'),
            num_labels=2
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    else:
        print(f"Model not found at {model_path}. Using a demo model.")
        model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    
    model.eval()
    
    # Load test data
    print("Loading test data...")
    try:
        fake_df = pd.read_csv(Path(data_path) / 'Fake.csv')
        true_df = pd.read_csv(Path(data_path) / 'True.csv')
        fake_df['label'] = 0
        true_df['label'] = 1
        df = pd.concat([fake_df, true_df], ignore_index=True)
        print(f"Loaded {len(df)} samples")
    except:
        print("Creating sample test data...")
        # Create sample data
        np.random.seed(42)
        test_texts = [
            "Scientists discover breakthrough in cancer research",
            "Breaking: Miracle cure that doctors hate!",
            "Government announces new climate policy",
            "You won't believe what this celebrity did!",
            "Researchers publish peer-reviewed study",
            "SHOCKING secret exposed by whistleblower!"
        ] * 50
        test_labels = [1, 0, 1, 0, 1, 0] * 50
        df = pd.DataFrame({'text': test_texts[:200], 'label': test_labels[:200]})
    
    return model, tokenizer, df


def evaluate_model(model, tokenizer, df, max_length=128, batch_size=32):
    """Evaluate model on dataset"""
    from torch.utils.data import DataLoader, Dataset
    
    class SimpleDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_length):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length
        
        def __len__(self):
            return len(self.texts)
        
        def __getitem__(self, idx):
            encoding = self.tokenizer(
                self.texts[idx],
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(self.labels[idx], dtype=torch.long)
            }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    dataset = SimpleDataset(df['text'].values, df['label'].values, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\nEvaluating on {len(df)} samples...")
    print(f"Using device: {device}")
    
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='binary', zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_probs[:, 1])
    }
    
    return metrics, y_true, y_pred, y_probs


def plot_results(y_true, y_pred, y_probs, save_path='../model/'):
    """Generate evaluation plots"""
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Fake', 'Real'], 
                yticklabels=['Fake', 'Real'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path / 'confusion_matrix.png', dpi=150)
    plt.close()
    
    # ROC Curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc_score(y_true, y_probs[:, 1]):.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path / 'roc_curve.png', dpi=150)
    plt.close()
    
    print(f"✓ Plots saved to {save_path}")


def main():
    print("="*60)
    print("FAKE NEWS DETECTION - MODEL EVALUATION")
    print("="*60)
    
    # Load model and data
    model, tokenizer, df = load_model_and_data()
    
    # Evaluate
    metrics, y_true, y_pred, y_probs = evaluate_model(model, tokenizer, df)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    for metric, value in metrics.items():
        print(f"{metric.upper():15s}: {value:.4f}")
    
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=['Fake', 'Real'], zero_division=0))
    
    # Generate plots
    plot_results(y_true, y_pred, y_probs)
    
    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()