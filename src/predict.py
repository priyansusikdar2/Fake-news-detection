"""
Prediction script for fake news detection
Run with: python predict.py
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')


class FakeNewsPredictor:
    """Simple predictor for fake news detection"""
    
    def __init__(self, model_path='../model/best_model.pt'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model and tokenizer
        print(f"Loading model from {model_path}...")
        
        if Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                checkpoint.get('model_name', 'distilbert-base-uncased'),
                num_labels=2
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print(f"Model not found. Using a demo model...")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                'distilbert-base-uncased', 
                num_labels=2
            )
        
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print("✓ Model loaded successfully!")
    
    def predict(self, text, max_length=128):
        """Predict if a news article is fake or real"""
        # Tokenize
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item()
        
        return {
            'label': 'Fake' if pred == 0 else 'Real',
            'confidence': confidence,
            'fake_probability': probs[0][0].item(),
            'real_probability': probs[0][1].item()
        }
    
    def predict_batch(self, texts, max_length=128):
        """Predict multiple texts"""
        results = []
        for text in texts:
            results.append(self.predict(text, max_length))
        return results


def interactive_mode(predictor):
    """Run interactive prediction mode"""
    print("\n" + "="*60)
    print("INTERACTIVE PREDICTION MODE")
    print("="*60)
    print("Enter a news article to classify (or 'quit' to exit)")
    print("-" * 60)
    
    while True:
        text = input("\n📰 Enter news text: ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not text:
            print("Please enter some text.")
            continue
        
        # Make prediction
        result = predictor.predict(text)
        
        # Display result
        print("\n" + "-" * 40)
        if result['label'] == 'Fake':
            print(f"⚠️  Prediction: {result['label']} NEWS")
            print(f"   Confidence: {result['confidence']:.2%}")
            print(f"   Fake probability: {result['fake_probability']:.2%}")
            print(f"   Real probability: {result['real_probability']:.2%}")
        else:
            print(f"✓ Prediction: {result['label']} NEWS")
            print(f"   Confidence: {result['confidence']:.2%}")
            print(f"   Fake probability: {result['fake_probability']:.2%}")
            print(f"   Real probability: {result['real_probability']:.2%}")
        print("-" * 40)


def demo_mode(predictor):
    """Run demo mode with sample texts"""
    print("\n" + "="*60)
    print("DEMO MODE - Sample Predictions")
    print("="*60)
    
    sample_texts = [
        "Scientists have discovered a new breakthrough in cancer research. The study, published in a peer-reviewed journal, shows promising results for new treatments.",
        "SHOCKING: Government hiding evidence of alien contact! Whistleblower reveals惊天秘密 that officials don't want you to know!",
        "The Federal Reserve announced a 0.25% interest rate increase today, citing concerns about inflation. Markets reacted moderately to the news.",
        "Miracle weight loss pill discovered! Doctors hate this simple trick that melts fat overnight! Click here to learn the secret!",
        "Local community organizes food drive for families in need during the holiday season. Over 500 families expected to benefit from the donations.",
        "YOU WON'T BELIEVE what this celebrity did at the airport! Exclusive photos inside!!"
    ]
    
    for i, text in enumerate(sample_texts, 1):
        print(f"\n{i}. {text[:100]}...")
        result = predictor.predict(text)
        print(f"   → {result['label']} (confidence: {result['confidence']:.2%})")


def main():
    print("="*60)
    print("FAKE NEWS DETECTION - PREDICTION SYSTEM")
    print("="*60)
    
    # Initialize predictor
    predictor = FakeNewsPredictor(model_path='../model/best_model.pt')
    
    # Choose mode
    print("\nSelect mode:")
    print("1. Demo mode (test with sample articles)")
    print("2. Interactive mode (enter your own articles)")
    print("3. Quick test")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == '1':
        demo_mode(predictor)
    elif choice == '2':
        interactive_mode(predictor)
    else:
        # Quick test
        test_text = "Scientists have discovered a new breakthrough in cancer research."
        result = predictor.predict(test_text)
        print(f"\nTest: '{test_text}'")
        print(f"Prediction: {result['label']} (confidence: {result['confidence']:.2%})")


if __name__ == "__main__":
    main()