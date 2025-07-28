#!/usr/bin/env python3
"""
Working PDF Extractor - Fixed for your LTC form PDF
This addresses the "No text blocks found" issue from your original code
"""

import os
import json
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import lightgbm as lgb
import re
from dataclasses import dataclass

@dataclass
class TextBlock:
    text: str
    font_size: float
    is_bold: bool
    is_italic: bool
    x: float
    y: float
    page: int
    font_family: str = ""

class WorkingPDFExtractor:
    """PDF extractor that actually works with your LTC form"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.ml_model = None
        self.trained = False
        self.label_encoder = {'title': 4, 'H1': 3, 'H2': 2, 'H3': 1, 'body': 0}
        self.reverse_label_encoder = {v: k for k, v in self.label_encoder.items()}
        
        # Try to load model if available
        if model_path and os.path.exists(model_path):
            print(f"🔄 Loading model from: {model_path}")
            self.train_model(model_path)
        else:
            print("⚠️  No model found, will use rule-based classification")
    
    def extract_text_blocks(self, pdf_path: str) -> List[TextBlock]:
        """Extract text blocks from PDF - FIXED VERSION"""
        print(f"🔄 Extracting text from: {pdf_path}")
        
        if not os.path.exists(pdf_path):
            print(f"❌ File not found: {pdf_path}")
            return []
        
        text_blocks = []
        
        try:
            doc = fitz.open(pdf_path)
            print(f"📄 PDF has {doc.page_count} page(s)")
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                print(f"   Processing page {page_num + 1}...")
                
                # Get detailed text blocks
                blocks = page.get_text("dict")
                page_block_count = 0
                
                for block in blocks["blocks"]:
                    if "lines" in block:  # This is a text block
                        for line in block["lines"]:
                            # Process each span (text with same formatting)
                            line_text_parts = []
                            line_spans = []
                            
                            for span in line["spans"]:
                                text = span["text"].strip()
                                if text:  # Only process non-empty text
                                    line_text_parts.append(text)
                                    line_spans.append(span)
                            
                            # Combine spans from the same line
                            if line_text_parts:
                                combined_text = " ".join(line_text_parts)
                                
                                # Use the formatting from the first span (or dominant span)
                                primary_span = line_spans[0]
                                font_size = primary_span.get("size", 12.0)
                                flags = primary_span.get("flags", 0)
                                is_bold = bool(flags & 2**4)  # Bold flag
                                is_italic = bool(flags & 2**1)  # Italic flag
                                bbox = primary_span.get("bbox", [0, 0, 0, 0])
                                font_family = primary_span.get("font", "")
                                
                                # Create text block
                                text_block = TextBlock(
                                    text=combined_text,
                                    font_size=font_size,
                                    is_bold=is_bold,
                                    is_italic=is_italic,
                                    x=bbox[0],
                                    y=bbox[1],
                                    page=page_num + 1,
                                    font_family=font_family
                                )
                                
                                text_blocks.append(text_block)
                                page_block_count += 1
                
                print(f"   ✅ Found {page_block_count} text blocks on page {page_num + 1}")
            
            doc.close()
            
        except Exception as e:
            print(f"❌ Error extracting text: {e}")
            return []
        
        print(f"✅ Total extracted: {len(text_blocks)} text blocks")
        
        # Show sample blocks for debugging
        if text_blocks:
            print(f"\n📝 Sample text blocks:")
            for i, block in enumerate(text_blocks[:5]):
                print(f"   {i+1}. '{block.text}' (size: {block.font_size:.1f}, bold: {block.is_bold})")
        
        return text_blocks
    
    def train_model(self, dataset_path: str):
        """Train the ML model"""
        try:
            df = pd.read_csv(dataset_path)
            print(f"📊 Training with {len(df)} samples")
            
            # Extract features
            features_list = []
            labels_list = []
            
            for _, row in df.iterrows():
                try:
                    text = str(row['text'])
                    font_size = float(row['font_size'])
                    is_bold = bool(row['is_bold'])
                    
                    # Create feature vector
                    features = [
                        font_size,
                        float(is_bold),
                        len(text.split()),  # word count
                        len(text),  # char count
                        float(text.istitle()),  # title case
                        float(text.isupper() and len(text) > 3),  # all caps
                        float(bool(re.search(r'\d', text))),  # has numbers
                        float(bool(re.match(r'^\d', text))),  # starts with number
                        sum(c.isupper() for c in text) / len(text) if text else 0  # caps ratio
                    ]
                    
                    features_list.append(features)
                    
                    # Encode label
                    label = str(row['label']).strip().lower()
                    encoded_label = self.label_encoder.get(label, 0)
                    labels_list.append(encoded_label)
                    
                except Exception:
                    continue
            
            if features_list:
                X = np.array(features_list)
                y = np.array(labels_list)
                
                # Train LightGBM
                train_data = lgb.Dataset(X, label=y)
                params = {
                    'objective': 'multiclass',
                    'num_class': 5,
                    'metric': 'multi_logloss',
                    'boosting_type': 'gbdt',
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'verbose': -1
                }
                
                self.ml_model = lgb.train(params, train_data, num_boost_round=100)
                self.trained = True
                print("✅ Model training completed")
                
        except Exception as e:
            print(f"⚠️  Model training failed: {e}")
            print("Will use rule-based classification instead")
    
    def classify_headings(self, text_blocks: List[TextBlock]) -> List[Dict]:
        """Classify text blocks as headings"""
        if not text_blocks:
            return []
        
        # Use ML model if available, otherwise use rules
        if self.trained and self.ml_model:
            return self._ml_classify(text_blocks)
        else:
            return self._rule_based_classify(text_blocks)
    
    def _ml_classify(self, text_blocks: List[TextBlock]) -> List[Dict]:
        """ML-based classification"""
        headings = []
        
        for block in text_blocks:
            try:
                # Extract features
                text = block.text
                features = [
                    block.font_size,
                    float(block.is_bold),
                    len(text.split()),
                    len(text),
                    float(text.istitle()),
                    float(text.isupper() and len(text) > 3),
                    float(bool(re.search(r'\d', text))),
                    float(bool(re.match(r'^\d', text))),
                    sum(c.isupper() for c in text) / len(text) if text else 0
                ]
                
                # Predict
                prediction = self.ml_model.predict([features])[0]
                predicted_class = np.argmax(prediction)
                confidence = float(np.max(prediction))
                
                level = self.reverse_label_encoder.get(predicted_class, 'body')
                
                if level != 'body' and confidence > 0.5:
                    headings.append({
                        'level': level,
                        'text': text,
                        'page': block.page,
                        'confidence': confidence
                    })
                    
            except Exception:
                continue
        
        return headings
    
    def _rule_based_classify(self, text_blocks: List[TextBlock]) -> List[Dict]:
        """Rule-based classification for when ML model isn't available"""
        headings = []
        
        # Calculate font size statistics
        font_sizes = [block.font_size for block in text_blocks]
        if not font_sizes:
            return []
        
        avg_font = np.mean(font_sizes)
        max_font = max(font_sizes)
        
        for block in text_blocks:
            text = block.text.strip()
            font_size = block.font_size
            
            # Skip very short text
            if len(text) < 2:
                continue
            
            level = 'body'
            confidence = 0.0
            
            # Title detection (largest font on first page)
            if (font_size == max_font and 
                block.page == 1 and 
                len(text.split()) <= 10):
                level = 'title'
                confidence = 0.9
            
            # H1 detection (large font or numbered sections)
            elif (font_size > avg_font * 1.2 or 
                  re.match(r'^\d+\.?\s+[A-Za-z]', text)):
                level = 'H1'
                confidence = 0.8
            
            # H2 detection (slightly larger font or subsections)
            elif (font_size > avg_font * 1.1 or 
                  re.match(r'^\d+\.\d+', text) or
                  (block.is_bold and len(text.split()) <= 8)):
                level = 'H2'
                confidence = 0.7
            
            # H3 detection (bold short text)
            elif (block.is_bold and len(text.split()) <= 5):
                level = 'H3'
                confidence = 0.6
            
            if level != 'body' and confidence > 0.5:
                headings.append({
                    'level': level,
                    'text': text,
                    'page': block.page,
                    'confidence': confidence
                })
        
        return headings
    
    def extract_outline(self, pdf_path: str) -> Dict:
        """Main extraction function"""
        print(f"\n📄 Processing: {pdf_path}")
        
        # Extract text blocks
        text_blocks = self.extract_text_blocks(pdf_path)
        
        if not text_blocks:
            return {"title": "", "outline": []}
        
        # Classify headings
        headings = self.classify_headings(text_blocks)
        
        # Separate title and outline
        title = ""
        outline = []
        
        for heading in headings:
            if heading['level'] == 'title' and not title:
                title = heading['text']
            elif heading['level'] != 'title':
                outline.append({
                    'level': heading['level'],
                    'text': heading['text'],
                    'page': heading['page']
                })
        
        # If no title found, use the first large text block
        if not title and text_blocks:
            for block in text_blocks:
                if (block.page == 1 and 
                    len(block.text.split()) <= 10 and
                    len(block.text) > 10):
                    title = block.text
                    break
        
        result = {
            "title": title.strip(),
            "outline": outline
        }
        
        print(f"✅ Extraction complete!")
        print(f"   📝 Title: '{title}'")
        print(f"   📋 Outline items: {len(outline)}")
        
        return result

def test_working_extractor():
    """Test the working extractor"""
    print("🚀 Testing Working PDF Extractor")
    print("="*50)
    
    # Find PDFs
    pdf_files = [f for f in os.listdir('.') if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print("❌ No PDF files found")
        return
    
    # Check for model
    model_path = None
    possible_models = [
        "heading_training_dataset_enhanced_fixed.csv",
        "heading_training_dataset_enhanced.csv"
    ]
    
    for model_file in possible_models:
        if os.path.exists(model_file):
            model_path = model_file
            break
    
    # Create extractor
    extractor = WorkingPDFExtractor(model_path)
    
    # Process each PDF
    results = {}
    
    for pdf_file in pdf_files:
        print(f"\n{'='*60}")
        result = extractor.extract_outline(pdf_file)
        results[pdf_file] = result
        
        # Save individual result
        output_file = f"result_{os.path.splitext(pdf_file)[0]}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Result saved to: {output_file}")
    
    # Save all results
    with open("all_extraction_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n🎉 All done! Results saved to all_extraction_results.json")

if __name__ == "__main__":
    test_working_extractor()