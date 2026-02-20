"""
Sentiment Analysis Module using FinBERT-PT-BR

This module provides functions to analyze sentiment in Brazilian financial news
using the FinBERT-PT-BR model (lucas-leme/FinBERT-PT-BR).

Based on: Santos et al. (2023) - FinBERT-PT-BR research paper
"""

import torch
import torch.nn.functional as F
import pandas as pd
from typing import List, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import os


def setup_sentiment_model(
    model_name: str = "lucas-leme/FinBERT-PT-BR"
) -> Tuple[AutoTokenizer, AutoModelForSequenceClassification, torch.device]:
    """
    Loads the FinBERT-PT-BR tokenizer and model.
    
    Args:
        model_name: HuggingFace model identifier
        
    Returns:
        Tuple of (tokenizer, model, device)
    """
    print(f"🔄 Loading model: {model_name}...")
    
    # Detect GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Device: {device}")
    
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
    
    # Load tokenizer and model
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model with safetensors to avoid torch security warning (CVE-2025-32434)
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            use_safetensors=True,
            weights_only=True
        )
    except Exception as e:
        print(f"⚠️ Failed to load with safetensors: {e}")
        print("   Falling back to standard loading (might fail if torch < 2.6)")
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # Move model to GPU if available
    model.to(device)
    model.eval()  # Set to evaluation mode
    
    print(f"✅ Model loaded successfully!")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return tokenizer, model, device


def predict_sentiment(
    text: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    device: torch.device,
    max_length: int = 512
) -> List[float]:
    """
    Predicts sentiment probabilities for a single text.
    
    Args:
        text: Input text to analyze
        tokenizer: FinBERT tokenizer
        model: FinBERT model
        device: torch device (cuda/cpu)
        max_length: Maximum token length (default: 512)
        
    Returns:
        List of probabilities: [prob_negative, prob_neutral, prob_positive]
    """
    # Handle empty or invalid text
    if not text or not isinstance(text, str):
        return [0.33, 0.34, 0.33]  # Return neutral probabilities
    
    # Tokenize input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True
    )
    
    # Move inputs to the same device as model
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    # Inference (no gradient computation)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Apply softmax to get probabilities
    probs = F.softmax(outputs.logits, dim=-1)
    probs_list = probs[0].cpu().tolist()
    
    # Model mapping: {0: 'POSITIVE', 1: 'NEGATIVE', 2: 'NEUTRAL'}
    # We want to return: [negative, neutral, positive]
    prob_pos = probs_list[0]
    prob_neg = probs_list[1]
    prob_neu = probs_list[2]
    
    # Return as list: [negative, neutral, positive]
    return [prob_neg, prob_neu, prob_pos]


def predict_batch(
    texts: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    device: torch.device,
    batch_size: int = 32,
    max_length: int = 512,
    show_progress: bool = True
) -> pd.DataFrame:
    """
    Predicts sentiment for a batch of texts with progress tracking.
    
    Args:
        texts: List of texts to analyze
        tokenizer: FinBERT tokenizer
        model: FinBERT model
        device: torch device
        batch_size: Number of texts to process at once
        max_length: Maximum token length
        show_progress: Whether to show progress bar
        
    Returns:
        DataFrame with columns: prob_neg, prob_neu, prob_pos, sentiment_score
    """
    results = []
    
    # Create progress bar
    iterator = range(0, len(texts), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Analyzing sentiment", unit="batch")
    
    for i in iterator:
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize batch
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True
        )
        
        # Move to device
        inputs = {key: val.to(device) for key, val in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get probabilities
        probs = F.softmax(outputs.logits, dim=-1)
        batch_probs = probs.cpu().tolist()
        
        # Model mapping: {0: 'POSITIVE', 1: 'NEGATIVE', 2: 'NEUTRAL'}
        # Reorder to: [negative, neutral, positive]
        for p in batch_probs:
            # p[0]=Pos, p[1]=Neg, p[2]=Neu
            results.append([p[1], p[2], p[0]])
    
    # Convert to DataFrame
    df_results = pd.DataFrame(
        results,
        columns=['prob_neg', 'prob_neu', 'prob_pos']
    )
    
    # Calculate sentiment score: prob_pos - prob_neg (range: -1 to 1)
    df_results['sentiment_score'] = df_results['prob_pos'] - df_results['prob_neg']
    
    return df_results


def calculate_sentiment_score(prob_neg: float, prob_pos: float) -> float:
    """
    Calculates a single sentiment score from probabilities.
    
    Formula: score = prob_positive - prob_negative
    
    Args:
        prob_neg: Probability of negative sentiment
        prob_pos: Probability of positive sentiment
        
    Returns:
        Sentiment score in range [-1, 1]
        -1 = very negative
         0 = neutral
        +1 = very positive
    """
    return prob_pos - prob_neg


def analyze_news_file(
    input_csv_path: str,
    output_csv_path: str,
    text_column: str = 'title',
    batch_size: int = 32,
    save_intermediate: bool = True,
    intermediate_interval: int = 1000
) -> pd.DataFrame:
    """
    Analyzes sentiment for all news in a CSV file.
    
    Args:
        input_csv_path: Path to input CSV with news data
        output_csv_path: Path to save results
        text_column: Column name containing text to analyze
        batch_size: Batch size for processing
        save_intermediate: Whether to save intermediate results
        intermediate_interval: Save every N rows
        
    Returns:
        DataFrame with original data + sentiment columns
    """
    print(f"\n{'='*60}")
    print("SENTIMENT ANALYSIS PIPELINE")
    print(f"{'='*60}\n")
    
    # Load data
    print(f"📂 Loading data from: {input_csv_path}")
    df = pd.read_csv(input_csv_path)
    print(f"   Total news: {len(df):,}")
    
    # Setup model
    tokenizer, model, device = setup_sentiment_model()
    
    # Extract texts
    texts = df[text_column].astype(str).tolist()
    
    # Predict sentiment
    print(f"\n🤖 Processing {len(texts):,} texts...")
    print(f"   Batch size: {batch_size}")
    print(f"   Device: {device}\n")
    
    sentiment_df = predict_batch(
        texts=texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=batch_size
    )
    
    # Merge with original data
    result_df = pd.concat([df, sentiment_df], axis=1)
    
    # Save results
    print(f"\n💾 Saving results to: {output_csv_path}")
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    result_df.to_csv(output_csv_path, index=False)
    
    print(f"✅ Analysis complete!")
    print(f"\n{'='*60}")
    
    return result_df


def main():
    """
    Example usage: Analyze sentiment for a sample of news.
    """
    # Test with sample texts
    sample_texts = [
        "Ibovespa sobe 2% com otimismo dos investidores após aprovação de reformas",
        "Bolsa desaba 3% com crise política e Banco Central eleva juros para conter inflação",
        "Mercado opera estável aguardando divulgação de dados econômicos desta semana",
        "Dólar dispara com tensão no cenário internacional e fuga de capital estrangeiro",
        "Ações da Petrobras sobem após anúncio de dividendos extraordinários"
    ]
    
    print("Loading model for testing...")
    tokenizer, model, device = setup_sentiment_model()
    
    print("\n" + "="*70)
    print("TESTING SENTIMENT ANALYSIS")
    print("="*70 + "\n")
    
    for i, text in enumerate(sample_texts, 1):
        probs = predict_sentiment(text, tokenizer, model, device)
        score = calculate_sentiment_score(probs[0], probs[2])
        
        print(f"Text {i}:")
        print(f"  '{text}'")
        print(f"  Negative: {probs[0]:.3f} | Neutral: {probs[1]:.3f} | Positive: {probs[2]:.3f}")
        print(f"  Score: {score:+.3f}")
        print()


if __name__ == "__main__":
    main()
