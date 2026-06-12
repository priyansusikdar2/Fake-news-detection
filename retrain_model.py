"""
Retrain model with balanced data to fix fake news detection
Run this script once to fix the model
"""

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
import os

print("="*60)
print("RETRAINING MODEL TO FIX FAKE NEWS DETECTION")
print("="*60)

# Create BALANCED dataset with clear fake news examples
fake_news_samples = [
    "BREAKING: Miracle cure discovered that doctors hate! Share before deleted!",
    "SHOCKING: Government hiding alien evidence from public! Whistleblower exposed!",
    "URGENT: You won't believe what this politician said on hidden camera!",
    "EXPOSED: Secret society controls world economy - leaked documents prove it!",
    "WARNING: Vaccine microchips being implanted in citizens right now!",
    "Scientists admit global warming is a complete HOAX! Truth revealed!",
    "Celebrity death hoax - famous actor found dead in hotel room!",
    "Financial secret that banks don't want you to know about! Get rich now!",
    "Natural disaster predicted for next week - government covering up!",
    "Ancient alien technology found under pyramid - archaeologists silenced!",
    "COVID vaccine killed more people than virus - study proves!",
    "5G towers activating mind control chips tomorrow! Prepare now!",
    "Bill Gates admits plan to depopulate world - leaked audio!",
    "Chemtrails are real - government spraying toxic chemicals!",
    "Moon landing was faked - NASA whistleblower finally confesses!",
    "Earth is flat - NASA admits hiding the truth for decades!",
    "Cloning technology used to replace world leaders - proof inside!",
    "Time travel device invented - government seized technology!",
    "Zombie virus escaped from lab - officials lying about outbreak!",
    "Secret tunnel found under White House - evidence of alien base!",
] * 50  # Create 1000 fake samples

real_news_samples = [
    "Scientists publish peer-reviewed study on climate change impacts in Nature journal.",
    "Federal Reserve announces 0.25% interest rate increase citing inflation concerns.",
    "Researchers discover new treatment for heart disease in clinical trial.",
    "Local community organizes food drive for 500 families in need this holiday season.",
    "Company reports quarterly earnings meeting analyst expectations for third quarter.",
    "University study shows 30 minutes of daily exercise reduces heart disease risk.",
    "City council votes 7-2 to approve new public transportation funding initiative.",
    "International summit discusses trade agreements between member nations.",
    "Medical association releases updated guidelines for diabetes treatment.",
    "Technology conference features innovative startup presentations from 50 companies.",
    "NASA successfully launches mission to study asteroid composition.",
    "WHO announces progress in reducing childhood malnutrition globally.",
    "New solar panel technology increases efficiency by 25% according to study.",
    "School district announces STEM program for underprivileged students.",
    "Police department releases statement on community safety initiatives.",
    "Hospital receives accreditation for excellence in patient care.",
    "Library announces free computer literacy classes for seniors.",
    "Park district opens new walking trails for community recreation.",
    "Fire department urges residents to check smoke detector batteries.",
    "Department of transportation announces road maintenance schedule.",
] * 50  # Create 1000 real samples

# Create DataFrame
df_fake = pd.DataFrame({'text': fake_news_samples, 'label': 0})
df_real = pd.DataFrame({'text': real_news_samples, 'label': 1})
df = pd.concat([df_fake, df_real], ignore_index=True)

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nDataset created:")
print(f"  Total samples: {len(df)}")
print(f"  Fake news: {sum(df['label']==0)}")
print(f"  Real news: {sum(df['label']==1)}")

# Convert to HuggingFace Dataset
dataset = Dataset.from_pandas(df)
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

def tokenize_function(examples):
    return tokenizer(
        examples['text'], 
        truncation=True, 
        padding='max_length', 
        max_length=128
    )

print("\nTokenizing dataset...")
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Split into train and test
train_test = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = train_test['train']
test_dataset = train_test['test']

print(f"Train samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# Load model
print("\nLoading model...")
model = AutoModelForSequenceClassification.from_pretrained(
    'distilbert-base-uncased', 
    num_labels=2
)

# Training arguments
training_args = TrainingArguments(
    output_dir='./training_results',
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
)

# Train
print("\n" + "="*60)
print("STARTING TRAINING...")
print("="*60)
trainer.train()

# Save model
print("\nSaving model...")
model.save_pretrained('model/fixed_model')
tokenizer.save_pretrained('model/fixed_model')
print("✓ Model saved to 'model/fixed_model'")

# Test the model
print("\n" + "="*60)
print("TESTING THE RETRAINED MODEL")
print("="*60)

model.eval()
test_cases = [
    ("BREAKING: Miracle cure discovered that doctors hate!", 0),
    ("Scientists publish peer-reviewed study.", 1),
    ("SHOCKING: Government hiding alien evidence!", 0),
    ("Federal Reserve announces interest rate decision.", 1),
]

for text, expected in test_cases:
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    
    label = "FAKE" if pred == 0 else "REAL"
    expected_label = "FAKE" if expected == 0 else "REAL"
    status = "✓" if pred == expected else "✗"
    
    print(f"\n{status} Text: {text[:60]}...")
    print(f"   Prediction: {label} (fake: {probs[0][0]:.3f}, real: {probs[0][1]:.3f})")
    print(f"   Expected: {expected_label}")

print("\n" + "="*60)
print("RETRAINING COMPLETE!")
print("="*60)
print("\nNow update your app.py to use the new model:")
print("Change MODEL_PATH to: 'model/fixed_model'")