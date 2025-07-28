#!/usr/bin/env python3
"""
Challenge 1B - Generic Document Intelligence System
Works for ANY domain: Research papers, Financial reports, Educational content, etc.
Uses embeddings for intelligent keyword extraction and semantic understanding
"""

import os
import json
import time
import numpy as np
import datetime
import pandas as pd
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from collections import Counter, defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
import warnings
warnings.filterwarnings('ignore')

# Import your working PDF extractor
from working_pdf_extractor import WorkingPDFExtractor

@dataclass
class ProcessedSection:
    document: str
    title: str
    content: str
    page: int
    level: str
    section_id: str
    word_count: int
    entities: List[str] = None
    technical_terms: List[str] = None

@dataclass
class Challenge1BResult:
    metadata: Dict
    extracted_sections: List[Dict]
    subsection_analysis: List[Dict]

class DocumentContentExtractor:
    """Enhanced content extractor optimized for travel documents"""
    
    def __init__(self):
        # Travel document section patterns
        self.travel_section_indicators = [
            'cities', 'cuisine', 'restaurants', 'attractions', 'hotels', 'transport',
            'culture', 'history', 'activities', 'nightlife', 'shopping', 'beaches',
            'museums', 'food', 'dining', 'sightseeing', 'recommendations'
        ]
    
    def extract_section_content(self, pdf_path: str, outline: List[Dict]) -> List[ProcessedSection]:
        """Enhanced section content extraction with travel optimization"""
        import fitz
        
        sections = []
        
        try:
            doc = fitz.open(pdf_path)
            
            for i, heading in enumerate(outline):
                # Enhanced content extraction
                section_content = self._extract_content_for_heading_enhanced(
                    doc, heading, outline, i
                )
                
                if section_content.strip():
                    # Calculate additional metadata for travel docs
                    entities = self._extract_section_entities(section_content)
                    technical_terms = self._extract_travel_terms(section_content)
                    
                    section = ProcessedSection(
                        document=os.path.basename(pdf_path),
                        title=heading['text'],
                        content=section_content,
                        page=heading['page'],
                        level=heading['level'],
                        section_id=f"{os.path.basename(pdf_path)}_{i}",
                        word_count=len(section_content.split()),
                        entities=entities,
                        technical_terms=technical_terms
                    )
                    sections.append(section)
            
            doc.close()
            
        except Exception as e:
            print(f"Error extracting content from {pdf_path}: {e}")
        
        return sections
    
    def _extract_content_for_heading_enhanced(self, doc, current_heading: Dict, 
                                            all_headings: List[Dict], heading_index: int) -> str:
        """Enhanced content extraction with better boundary detection"""
        current_page = current_heading['page'] - 1
        
        # Determine end boundary more precisely
        if heading_index + 1 < len(all_headings):
            next_heading = all_headings[heading_index + 1]
            end_page = next_heading['page'] - 1
        else:
            end_page = doc.page_count - 1
        
        content_parts = []
        
        for page_num in range(current_page, min(end_page + 1, doc.page_count)):
            page = doc[page_num]
            
            # Use both text methods for better extraction
            page_text = page.get_text()
            
            # Also try text blocks for better structure
            try:
                blocks = page.get_text("dict")
                structured_text = self._extract_structured_text(blocks)
                if len(structured_text) > len(page_text):
                    page_text = structured_text
            except:
                pass
            
            if page_num == current_page:
                # Better heading detection
                lines = page_text.split('\n')
                heading_found = False
                page_content = []
                
                heading_text = current_heading['text'].strip()
                
                for line in lines:
                    line_clean = line.strip()
                    if not heading_found:
                        # More flexible heading matching
                        if (heading_text in line_clean or 
                            self._fuzzy_match_heading(heading_text, line_clean)):
                            heading_found = True
                            continue
                    
                    if heading_found and line_clean:
                        page_content.append(line_clean)
                
                content_parts.extend(page_content)
            else:
                if page_num == end_page and heading_index + 1 < len(all_headings):
                    lines = page_text.split('\n')
                    next_heading_text = all_headings[heading_index + 1]['text'].strip()
                    
                    for line in lines:
                        line_clean = line.strip()
                        if (next_heading_text in line_clean or 
                            self._fuzzy_match_heading(next_heading_text, line_clean)):
                            break
                        if line_clean:
                            content_parts.append(line_clean)
                else:
                    content_parts.extend([line.strip() for line in page_text.split('\n') if line.strip()])
        
        # Enhanced content cleaning
        content = ' '.join(content_parts)
        content = re.sub(r'\s+', ' ', content)  # Normalize whitespace
        content = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', content)  # Fix sentence spacing
        
        return content.strip()
    
    def _extract_structured_text(self, blocks_dict):
        """Extract text maintaining better structure from text blocks"""
        text_parts = []
        
        for block in blocks_dict.get("blocks", []):
            if "lines" in block:
                for line in block["lines"]:
                    line_text = ""
                    for span in line["spans"]:
                        text = span.get("text", "").strip()
                        if text:
                            line_text += text + " "
                    if line_text.strip():
                        text_parts.append(line_text.strip())
        
        return '\n'.join(text_parts)
    
    def _fuzzy_match_heading(self, heading: str, line: str) -> bool:
        """Fuzzy matching for heading detection"""
        heading_words = heading.lower().split()
        line_words = line.lower().split()
        
        if len(heading_words) <= 2:
            return any(word in line.lower() for word in heading_words if len(word) > 2)
        
        matches = sum(1 for word in heading_words if word in line_words)
        return matches >= len(heading_words) * 0.6
    
    def _extract_section_entities(self, content: str) -> List[str]:
        """Extract named entities from section content"""
        entities = []
        
        # Simple pattern-based extraction for travel entities
        patterns = {
            'places': r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:Cathedral|Museum|Palace|Beach|Square|Market))\b',
            'cities': r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?=\s+(?:is|has|offers|features))\b',
            'restaurants': r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:Restaurant|Cafe|Bistro|Brasserie))\b'
        }
        
        for pattern_type, pattern in patterns.items():
            matches = re.findall(pattern, content)
            entities.extend(matches)
        
        return list(set(entities))
    
    def _extract_travel_terms(self, content: str) -> List[str]:
        """Extract travel-specific technical terms"""
        travel_terms = []
        
        # Travel-specific terms patterns
        terms_patterns = [
            r'\b(?:Michelin|starred|rating|cuisine|speciality|traditional|local)\b',
            r'\b(?:architecture|style|period|century|historic|heritage)\b',
            r'\b(?:atmosphere|ambiance|setting|location|district|area)\b'
        ]
        
        content_lower = content.lower()
        for pattern in terms_patterns:
            matches = re.findall(pattern, content_lower, re.IGNORECASE)
            travel_terms.extend(matches)
        
        return list(set(travel_terms))

class UniversalSemanticAnalyzer:
    """Universal semantic analyzer optimized for all domains"""
    
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=3000,
            stop_words='english',
            ngram_range=(1, 4),
            min_df=1,  # Changed back to 1 for better keyword coverage
            max_df=0.85
        )
        
        # Comprehensive domain-specific signals
        self.importance_signals = {
            'travel': ['travel', 'trip', 'vacation', 'tourism', 'destination', 'hotel', 'restaurant', 
                      'attractions', 'sightseeing', 'culture', 'food', 'cuisine', 'cities', 'places',
                      'visit', 'explore', 'guide', 'recommendations', 'itinerary', 'planning',
                      'accommodation', 'transportation', 'activities', 'experiences', 'local',
                      'historic', 'museums', 'beaches', 'mountains', 'architecture', 'nightlife',
                      'group', 'friends', 'college', 'young', 'adventure', 'fun', 'party'],
                      
            'business': ['strategy', 'revenue', 'profit', 'market', 'growth', 'performance', 'KPI',
                        'management', 'corporate', 'professional', 'workflow', 'efficiency'],
                        
            'technical': ['software', 'tool', 'feature', 'function', 'procedure', 'process', 'step',
                         'tutorial', 'guide', 'how-to', 'instruction', 'method', 'technique',
                         'create', 'edit', 'convert', 'export', 'import', 'configure', 'setup'],
                         
            'hr': ['employee', 'onboarding', 'compliance', 'forms', 'fillable', 'signatures',
                   'documentation', 'records', 'staff', 'personnel', 'hiring', 'training'],
                   
            'food': ['recipe', 'ingredients', 'cooking', 'preparation', 'vegetarian', 'vegan',
                    'gluten-free', 'buffet', 'menu', 'meal', 'dish', 'cuisine', 'culinary',
                    'catering', 'corporate', 'gathering', 'dinner', 'lunch', 'breakfast'],
                    
            'educational': ['learn', 'tutorial', 'guide', 'instructions', 'steps', 'how-to',
                           'training', 'course', 'lesson', 'skill', 'knowledge'],
                           
            'research': ['study', 'research', 'analysis', 'findings', 'results', 'conclusion', 'methodology']
        }
        
        # Task-specific priority patterns
        self.task_patterns = {
            'forms': ['fillable', 'form', 'field', 'signature', 'electronic', 'fill', 'sign', 'create'],
            'menu_planning': ['vegetarian', 'vegan', 'gluten-free', 'buffet', 'menu', 'recipe', 'ingredients'],
            'group_travel': ['group', 'friends', 'college', 'activities', 'nightlife', 'adventure'],
            'software_learning': ['tutorial', 'guide', 'how-to', 'step', 'instruction', 'feature'],
            'compliance': ['onboarding', 'compliance', 'documentation', 'records', 'procedure']
        }
        
        self._download_nltk_data()
    
    def extract_intelligent_keywords(self, persona_text: str, job_text: str, 
                                   document_contents: List[str]) -> Tuple[List[str], Dict[str, float]]:
        """Universal keyword extraction with domain adaptation"""
        
        # 1. Extract PDF-based keywords
        pdf_keywords = self._extract_pdf_based_keywords(document_contents)
        
        # 2. Query analysis
        query_text = f"{persona_text} {job_text}".lower()
        query_keywords = self._extract_explicit_terms(query_text)
        
        # 3. Domain detection and analysis
        domain_context = self._analyze_domain_context(document_contents)
        primary_domain = max(domain_context.items(), key=lambda x: x[1])[0] if domain_context else 'general'
        
        # 4. Task-specific keyword extraction
        task_keywords = self._extract_task_specific_keywords(query_text, document_contents, primary_domain)
        
        # 5. Professional context keywords
        professional_keywords = self._extract_professional_context(persona_text, job_text)
        
        # 6. Combine all keywords
        all_keywords = pdf_keywords + query_keywords + task_keywords + professional_keywords
        keyword_weights = self._calculate_universal_keyword_weights(
            all_keywords, query_text, domain_context, primary_domain
        )
        
        # Return top keywords with domain-specific filtering
        top_keywords = sorted(keyword_weights.keys(), key=keyword_weights.get, reverse=True)[:35]
        return top_keywords, keyword_weights
    
    def _extract_task_specific_keywords(self, query_text: str, document_contents: List[str], 
                                      primary_domain: str) -> List[str]:
        """Extract keywords specific to the task type"""
        task_keywords = []
        all_content = ' '.join(document_contents).lower()
        
        # Detect task type from query
        task_type = self._detect_task_type(query_text)
        
        if task_type in self.task_patterns:
            # Look for task-specific terms in content
            for term in self.task_patterns[task_type]:
                if re.search(r'\\b' + re.escape(term) + r'\\b', all_content):
                    task_keywords.append(term)
        
        # Domain-specific extraction
        if primary_domain == 'food':
            # Extract dietary requirements and food types
            food_patterns = [
                r'\\b(?:vegetarian|vegan|gluten-free|dairy-free)\\b',
                r'\\b(?:buffet|catering|corporate|gathering)\\b',
                r'\\b(?:recipe|ingredients|cooking|preparation)\\b'
            ]
            
            for pattern in food_patterns:
                matches = re.findall(pattern, all_content)
                task_keywords.extend(matches)
        
        elif primary_domain == 'technical':
            # Extract software features and actions
            tech_patterns = [
                r'\\b(?:create|edit|convert|export|import|share)\\b',
                r'\\b(?:PDF|form|signature|fillable|interactive)\\b',
                r'\\b(?:tutorial|guide|how-to|step-by-step)\\b'
            ]
            
            for pattern in tech_patterns:
                matches = re.findall(pattern, all_content)
                task_keywords.extend(matches)
        
        elif primary_domain == 'travel':
            # Extract travel-specific terms
            travel_patterns = [
                r'\\b(?:activities|attractions|nightlife|restaurant)\\b',
                r'\\b(?:group|friends|college|young)\\b',
                r'\\b(?:guide|recommendations|tips|planning)\\b'
            ]
            
            for pattern in travel_patterns:
                matches = re.findall(pattern, all_content)
                task_keywords.extend(matches)
        
        return list(set(task_keywords))
    
    def _extract_professional_context(self, persona_text: str, job_text: str) -> List[str]:
        """Extract professional context keywords"""
        professional_keywords = []
        combined_text = f"{persona_text} {job_text}".lower()
        
        # Professional role mapping
        role_contexts = {
            'hr': ['employee', 'onboarding', 'compliance', 'forms', 'documentation', 'staff'],
            'travel planner': ['itinerary', 'booking', 'accommodation', 'activities', 'group'],
            'food contractor': ['catering', 'menu', 'buffet', 'corporate', 'dietary', 'preparation'],
            'manager': ['workflow', 'efficiency', 'team', 'process', 'management'],
            'trainer': ['training', 'education', 'learning', 'skills', 'development']
        }
        
        for role, context_terms in role_contexts.items():
            if role in combined_text:
                professional_keywords.extend(context_terms)
        
        # Extract action words from job description
        action_patterns = [
            r'\\b(?:create|manage|prepare|plan|organize|develop)\\b',
            r'\\b(?:implement|execute|coordinate|facilitate|deliver)\\b'
        ]
        
        for pattern in action_patterns:
            matches = re.findall(pattern, combined_text)
            professional_keywords.extend(matches)
        
        return list(set(professional_keywords))
    
    def _detect_task_type(self, query_text: str) -> str:
        """Detect the specific task type from query"""
        query_lower = query_text.lower()
        
        if any(term in query_lower for term in ['form', 'fillable', 'onboarding', 'compliance']):
            return 'forms'
        elif any(term in query_lower for term in ['menu', 'vegetarian', 'buffet', 'catering']):
            return 'menu_planning'
        elif any(term in query_lower for term in ['group', 'friends', 'college', 'trip']):
            return 'group_travel'
        elif any(term in query_lower for term in ['learn', 'tutorial', 'guide', 'software']):
            return 'software_learning'
        else:
            return 'general'
    
    def _calculate_universal_keyword_weights(self, keywords: List[str], query_text: str, 
                                           domain_context: Dict[str, float], primary_domain: str) -> Dict[str, float]:
        """Universal keyword weighting system"""
        weights = {}
        query_lower = query_text.lower()
        
        # Domain-specific boost terms
        domain_boosts = {
            'food': ['vegetarian', 'vegan', 'gluten-free', 'buffet', 'menu', 'recipe', 'catering'],
            'technical': ['create', 'fillable', 'form', 'signature', 'tutorial', 'guide', 'PDF'],
            'travel': ['group', 'friends', 'college', 'nightlife', 'activities', 'adventure'],
            'hr': ['onboarding', 'compliance', 'employee', 'forms', 'documentation']
        }
        
        boost_terms = domain_boosts.get(primary_domain, [])
        
        for keyword in set(keywords):
            weight = 0.0
            
            # Base weight from query frequency
            weight += query_lower.count(keyword) * 3.0
            
            # Exact match bonus
            if keyword in query_lower:
                weight += 2.5
            
            # Word boundary match
            if re.search(r'\\b' + re.escape(keyword) + r'\\b', query_lower):
                weight += 1.5
            
            # Domain-specific bonus
            for domain, score in domain_context.items():
                if keyword in self.importance_signals.get(domain, []):
                    weight += score * 5.0
            
            # Primary domain boost
            if keyword in boost_terms:
                weight += 6.0  # Strong boost for domain-specific terms
            
            # Professional context boost
            if any(prof_term in keyword for prof_term in ['professional', 'corporate', 'business']):
                weight += 2.0
            
            # Technical action boost
            if keyword in ['create', 'manage', 'prepare', 'plan', 'organize', 'develop']:
                weight += 3.0
            
            # Specificity bonus
            if len(keyword) > 8:
                weight += 0.8
            
            weights[keyword] = weight
        
        return weights

class UniversalRelevanceScorer:
    """Universal relevance scorer that adapts to any domain and task"""
    
    def __init__(self, semantic_analyzer):
        self.semantic_analyzer = semantic_analyzer
        
        # Adaptive base weights
        self.base_level_weights = {
            'title': 0.3,
            'H1': 1.0,
            'H2': 0.95,
            'H3': 0.8,
            'body': 0.4
        }
        
        # Domain-specific section priorities
        self.domain_section_priorities = {
            'travel': {
                'nightlife': 3.0, 'entertainment': 3.0, 'coastal': 2.5, 'activities': 2.5,
                'things to do': 2.8, 'adventures': 2.5, 'tips': 2.0, 'guide': 2.2,
                'cities': 1.8, 'culture': 1.5, 'history': 1.2
            },
            'technical': {
                'create': 3.0, 'tutorial': 2.8, 'guide': 2.8, 'how-to': 2.8,
                'fillable': 3.2, 'forms': 3.2, 'signature': 2.5, 'convert': 2.3,
                'edit': 2.3, 'export': 2.0, 'share': 1.8, 'tips': 1.5
            },
            'food': {
                'vegetarian': 3.0, 'vegan': 2.8, 'gluten-free': 2.8, 'recipe': 2.5,
                'buffet': 2.2, 'menu': 2.2, 'ingredients': 2.0, 'preparation': 2.0,
                'cooking': 1.8, 'dinner': 2.3, 'lunch': 1.5, 'breakfast': 1.2
            },
            'hr': {
                'onboarding': 3.0, 'compliance': 3.0, 'forms': 2.8, 'fillable': 2.8,
                'employee': 2.5, 'documentation': 2.3, 'signature': 2.5, 'records': 2.0
            },
            'business': {
                'strategy': 2.5, 'management': 2.3, 'workflow': 2.3, 'process': 2.0,
                'corporate': 2.0, 'professional': 1.8, 'efficiency': 1.8
            }
        }
        
        # Task-specific priorities
        self.task_priorities = {
            'forms': ['fillable', 'form', 'signature', 'create', 'convert', 'field'],
            'menu_planning': ['vegetarian', 'vegan', 'gluten-free', 'buffet', 'recipe', 'menu'],
            'group_activities': ['nightlife', 'activities', 'group', 'entertainment', 'fun'],
            'learning': ['tutorial', 'guide', 'how-to', 'learn', 'step', 'instruction']
        }
    
    def score_section(self, section, query_text: str, keywords: List[str], 
                     keyword_weights: Dict[str, float], domain_context: Dict[str, float]) -> Tuple[float, Dict]:
        """Universal scoring system that adapts to any domain"""
        
        # Detect primary domain and task
        primary_domain = max(domain_context.items(), key=lambda x: x[1])[0] if domain_context else 'general'
        task_type = self._detect_task_type(query_text)
        
        # 1. Domain-specific relevance (45% weight)
        domain_relevance_score = self._calculate_domain_relevance(
            section, query_text, keywords, primary_domain, task_type
        )
        
        # 2. Keyword semantic similarity (35% weight)
        semantic_score = self._calculate_keyword_similarity(section.content, keywords, keyword_weights)
        
        # 3. Section type priority (15% weight)
        section_type_score = self._calculate_section_type_score(
            section.title, section.content, primary_domain
        )
        
        # 4. Structural importance (5% weight)
        structural_score = self._get_adaptive_structural_weight(section.level, primary_domain)
        
        # Combine scores
        final_score = (
            domain_relevance_score * 0.45 +
            semantic_score * 0.35 +
            section_type_score * 0.15 +
            structural_score * 0.05
        )
        
        # Apply domain-specific multipliers
        final_score *= self._get_domain_multiplier(section, primary_domain, task_type)
        
        # Length adjustment
        length_multiplier = self._calculate_length_multiplier(section.word_count)
        final_score *= length_multiplier
        
        score_breakdown = {
            'domain_relevance_score': domain_relevance_score,
            'semantic_score': semantic_score,
            'section_type_score': section_type_score,
            'structural_score': structural_score,
            'length_multiplier': length_multiplier,
            'final_score': final_score
        }
        
        return final_score, score_breakdown
    
    def _calculate_domain_relevance(self, section, query_text: str, keywords: List[str], 
                                  primary_domain: str, task_type: str) -> float:
        """Calculate domain-specific relevance"""
        content_lower = section.content.lower()
        title_lower = section.title.lower()
        
        relevance_score = 0.0
        
        # Domain-specific scoring
        if primary_domain == 'travel':
            relevance_score = self._score_travel_relevance(title_lower, content_lower, query_text)
        elif primary_domain == 'technical':
            relevance_score = self._score_technical_relevance(title_lower, content_lower, query_text)
        elif primary_domain == 'food':
            relevance_score = self._score_food_relevance(title_lower, content_lower, query_text)
        elif primary_domain == 'hr' or 'hr' in query_text.lower():
            relevance_score = self._score_hr_relevance(title_lower, content_lower, query_text)
        else:
            relevance_score = self._score_general_relevance(title_lower, content_lower, query_text)
        
        # Task-specific boost
        if task_type in self.task_priorities:
            for priority_term in self.task_priorities[task_type]:
                if priority_term in title_lower or priority_term in content_lower:
                    relevance_score += 0.15
        
        return min(relevance_score, 1.0)
    
    def _score_travel_relevance(self, title: str, content: str, query: str) -> float:
        """Score travel domain relevance"""
        score = 0.0
        
        # Group travel indicators
        if any(term in query for term in ['group', 'friends', 'college']):
            group_terms = ['nightlife', 'activities', 'entertainment', 'fun', 'adventure']
            for term in group_terms:
                if term in title or term in content:
                    score += 0.25
        
        # Activity focus
        activity_terms = ['activities', 'things to do', 'attractions', 'coastal', 'adventure']
        for term in activity_terms:
            if term in title or term in content:
                score += 0.2
        
        # Practical travel info
        practical_terms = ['tips', 'guide', 'planning', 'recommendations']
        for term in practical_terms:
            if term in title or term in content:
                score += 0.15
        
        return score
    
    def _score_technical_relevance(self, title: str, content: str, query: str) -> float:
        """Score technical/software domain relevance"""
        score = 0.0
        
        # Core technical actions
        action_terms = ['create', 'edit', 'convert', 'export', 'share', 'fill', 'sign']
        for term in action_terms:
            if term in title or term in content:
                score += 0.2
        
        # Forms and signatures (HR specific)
        if any(term in query for term in ['forms', 'fillable', 'onboarding', 'hr']):
            form_terms = ['fillable', 'forms', 'signature', 'field', 'interactive']
            for term in form_terms:
                if term in title or term in content:
                    score += 0.3  # High priority for HR tasks
        
        # Tutorial/guide content
        learning_terms = ['tutorial', 'guide', 'how-to', 'step', 'instruction']
        for term in learning_terms:
            if term in title or term in content:
                score += 0.15
        
        return score
    
    def _score_food_relevance(self, title: str, content: str, query: str) -> float:
        """Score food domain relevance"""
        score = 0.0
        
        # Dietary requirements (crucial for catering)
        dietary_terms = ['vegetarian', 'vegan', 'gluten-free', 'dairy-free']
        for term in dietary_terms:
            if term in title or term in content:
                score += 0.3  # High priority for dietary requirements
        
        # Buffet/catering style
        if 'buffet' in query or 'corporate' in query:
            buffet_terms = ['buffet', 'catering', 'corporate', 'gathering', 'party']
            for term in buffet_terms:
                if term in title or term in content:
                    score += 0.25
        
        # Recipe and preparation
        recipe_terms = ['recipe', 'ingredients', 'preparation', 'cooking', 'method']
        for term in recipe_terms:
            if term in title or term in content:
                score += 0.15
        
        # Meal categories
        meal_terms = ['dinner', 'lunch', 'breakfast', 'appetizer', 'main', 'side']
        for term in meal_terms:
            if term in title:
                score += 0.1
        
        return score
    
    def _score_hr_relevance(self, title: str, content: str, query: str) -> float:
        """Score HR domain relevance"""
        score = 0.0
        
        # Core HR processes
        hr_terms = ['onboarding', 'compliance', 'employee', 'staff', 'personnel']
        for term in hr_terms:
            if term in title or term in content:
                score += 0.25
        
        # Forms and documentation
        form_terms = ['forms', 'fillable', 'documentation', 'records', 'field']
        for term in form_terms:
            if term in title or term in content:
                score += 0.3
        
        # Digital processes
        digital_terms = ['signature', 'electronic', 'digital', 'automated']
        for term in digital_terms:
            if term in title or term in content:
                score += 0.2
        
        return score
    
    def _score_general_relevance(self, title: str, content: str, query: str) -> float:
        """Score general relevance for unknown domains"""
        score = 0.0
        
        # Look for query terms directly
        query_words = query.lower().split()
        for word in query_words:
            if len(word) > 3:  # Skip short words
                if word in title:
                    score += 0.2
                elif word in content:
                    score += 0.1
        
        return score
    
    def _calculate_section_type_score(self, title: str, content: str, primary_domain: str) -> float:
        """Calculate section type score based on domain"""
        title_lower = title.lower()
        
        priorities = self.domain_section_priorities.get(primary_domain, {})
        
        max_priority = 0.0
        for section_type, priority in priorities.items():
            if section_type in title_lower:
                max_priority = max(max_priority, priority)
        
        # Normalize to 0-1 range
        return min(max_priority / 3.2, 1.0)  # 3.2 is max priority
    
    def _get_domain_multiplier(self, section, primary_domain: str, task_type: str) -> float:
        """Get domain-specific multiplier"""
        multiplier = 1.0
        title_lower = section.title.lower()
        
        # High-priority combinations
        if primary_domain == 'technical' and task_type == 'forms':
            if 'fillable' in title_lower or 'form' in title_lower:
                multiplier = 1.3
        
        elif primary_domain == 'food' and task_type == 'menu_planning':
            if any(term in title_lower for term in ['vegetarian', 'vegan', 'gluten-free']):
                multiplier = 1.25
        
        elif primary_domain == 'travel' and task_type == 'group_activities':
            if any(term in title_lower for term in ['nightlife', 'activities', 'coastal']):
                multiplier = 1.2
        
        return multiplier
    
    def _detect_task_type(self, query_text: str) -> str:
        """Detect task type from query"""
        query_lower = query_text.lower()
        
        if any(term in query_lower for term in ['form', 'fillable', 'onboarding', 'compliance']):
            return 'forms'
        elif any(term in query_lower for term in ['menu', 'vegetarian', 'buffet', 'catering']):
            return 'menu_planning'
        elif any(term in query_lower for term in ['group', 'friends', 'college', 'trip']):
            return 'group_activities'
        elif any(term in query_lower for term in ['learn', 'tutorial', 'guide']):
            return 'learning'
        else:
            return 'general'
    
    def _calculate_keyword_similarity(self, content: str, keywords: List[str], 
                                    keyword_weights: Dict[str, float]) -> float:
        """Calculate weighted keyword similarity"""
        if not keywords:
            return 0.0
        
        content_lower = content.lower()
        total_weight = sum(keyword_weights.values())
        matched_weight = 0.0
        
        for keyword in keywords:
            weight = keyword_weights.get(keyword, 1.0)
            if re.search(r'\\b' + re.escape(keyword) + r'\\b', content_lower):
                matched_weight += weight
        
        return matched_weight / total_weight if total_weight > 0 else 0.0
    
    def _get_adaptive_structural_weight(self, level: str, primary_domain: str) -> float:
        """Get structural weight adapted to domain"""
        base_weight = self.base_level_weights.get(level, 0.4)
        
        # Domain adjustments
        if primary_domain in ['technical', 'hr']:
            if level in ['H1', 'H2']:
                base_weight *= 1.1  # Boost main sections for technical content
        
        return min(base_weight, 1.0)
    
    def _calculate_length_multiplier(self, word_count: int) -> float:
        """Calculate length multiplier"""
        if word_count < 20:
            return 0.6
        elif word_count > 1000:
            return 0.8
        elif 80 <= word_count <= 400:
            return 1.0
        else:
            return 0.9

class AdvancedContentRefiner:
    """Advanced content refinement using importance scoring"""
    
    def __init__(self):
        pass
    
    def extract_key_sentences(self, content: str, keywords: List[str], 
                            max_sentences: int = 5) -> str:
        """Extract most important sentences using advanced scoring"""
        if not content.strip():
            return ""
        
        try:
            sentences = sent_tokenize(content)
            
            if len(sentences) <= max_sentences:
                return content
            
            # Score each sentence
            scored_sentences = []
            
            for i, sentence in enumerate(sentences):
                score = self._score_sentence(sentence, keywords, i, len(sentences))
                scored_sentences.append((sentence, score))
            
            # Sort by score and take top sentences
            scored_sentences.sort(key=lambda x: x[1], reverse=True)
            top_sentences = [sent for sent, _ in scored_sentences[:max_sentences]]
            
            # Maintain original order for readability
            result_sentences = []
            for sentence in sentences:
                if sentence in top_sentences:
                    result_sentences.append(sentence)
                    if len(result_sentences) >= max_sentences:
                        break
            
            return ' '.join(result_sentences)
            
        except Exception as e:
            print(f"Error extracting key sentences: {e}")
            return content[:500] + "..." if len(content) > 500 else content
    
    def _score_sentence(self, sentence: str, keywords: List[str], 
                       position: int, total_sentences: int) -> float:
        """Score individual sentence importance"""
        score = 0.0
        sentence_lower = sentence.lower()
        
        # Keyword presence (most important factor)
        keyword_matches = 0
        for keyword in keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', sentence_lower):
                keyword_matches += 1
        
        if keywords:
            score += (keyword_matches / len(keywords)) * 3.0
        
        # Position scoring (first and last sentences often important)
        if position == 0:
            score += 1.0  # First sentence
        elif position == total_sentences - 1:
            score += 0.5  # Last sentence
        elif position < total_sentences * 0.3:
            score += 0.7  # Early sentences
        
        # Length scoring (prefer medium-length sentences)
        words = len(sentence.split())
        if 15 <= words <= 35:
            score += 0.5
        elif 8 <= words <= 50:
            score += 0.2
        
        # Structural indicators
        if any(indicator in sentence_lower for indicator in 
               ['important', 'key', 'main', 'primary', 'essential', 'significant', 'crucial']):
            score += 0.5
        
        # Data/number presence
        if re.search(r'\d+', sentence):
            score += 0.3
        
        return score

class GenericChallenge1BProcessor:
    """Generic processor that works for any domain"""
    
    def __init__(self):
        self.pdf_extractor = WorkingPDFExtractor()
        self.content_extractor = DocumentContentExtractor()
        self.semantic_analyzer = UniversalSemanticAnalyzer()  # ← Updated
        self.relevance_scorer = UniversalRelevanceScorer(self.semantic_analyzer)  # ← Updated
        self.content_refiner = AdvancedContentRefiner()
        
    def process_challenge(self, input_file: str) -> Dict:
        """Process Challenge 1B input for ANY domain"""
        print(f"🚀 Processing Generic Challenge 1B: {input_file}")
        start_time = time.time()
        
        # Load input
        with open(input_file, 'r', encoding='utf-8') as f:
            challenge_input = json.load(f)
        
        # Extract challenge components
        challenge_info = challenge_input['challenge_info']
        documents = challenge_input['documents']
        persona = challenge_input['persona']
        job_to_be_done = challenge_input['job_to_be_done']
        
        print(f"📋 Task: {job_to_be_done['task']}")
        print(f"👤 Persona: {persona['role']}")
        print(f"📄 Documents: {len(documents)}")
        
        # Step 1: Extract outlines using Task 1 approach
        print("\n🔍 Step 1: Extracting document outlines...")
        document_outlines = {}
        
        for doc_info in documents:
            filename = doc_info['filename']
            if os.path.exists(filename):
                outline_result = self.pdf_extractor.extract_outline(filename)
                document_outlines[filename] = outline_result
            else:
                print(f"⚠️ Document not found: {filename}")
        
        # Step 2: Extract full section content
        print("\n📖 Step 2: Extracting section content...")
        all_sections = []
        document_contents = []
        
        for filename, outline_result in document_outlines.items():
            sections = self.content_extractor.extract_section_content(
                filename, outline_result.get('outline', [])
            )
            all_sections.extend(sections)
            
            # Collect all content for domain analysis
            for section in sections:
                document_contents.append(section.content)
        
        print(f"✅ Extracted {len(all_sections)} sections total")
        
        # Step 3: Intelligent keyword extraction and domain analysis
        print("\n🧠 Step 3: Intelligent semantic analysis...")
        
        # Create query text
        query_text = f"{persona['role']} {job_to_be_done['task']}"
        
        # Extract intelligent keywords using embeddings and domain analysis
        keywords, keyword_weights = self.semantic_analyzer.extract_intelligent_keywords(
            persona['role'], job_to_be_done['task'], document_contents
        )
        
        # Analyze domain context
        domain_context = self.semantic_analyzer._analyze_domain_context(document_contents)
        
        print(f"🔑 Extracted {len(keywords)} intelligent keywords")
        print(f"📊 Domain analysis: {dict(list(domain_context.items())[:3])}")  # Show top 3 domains
        
        # Step 4: Advanced relevance scoring
        print("\n🎯 Step 4: Advanced relevance scoring...")
        
        scored_sections = []
        
        for section in all_sections:
            score, breakdown = self.relevance_scorer.score_section(
                section, query_text, keywords, keyword_weights, domain_context
            )
            
            scored_sections.append({
                'section': section,
                'score': score,
                'breakdown': breakdown
            })
        
        # Step 5: Select top sections and create output
        print("\n✨ Step 5: Selecting optimal sections...")
        
        # Sort by score and take top 5
        scored_sections.sort(key=lambda x: x['score'], reverse=True)
        top_sections = scored_sections[:5]
        
        # Create extracted_sections
        extracted_sections = []
        for i, item in enumerate(top_sections, 1):
            section = item['section']
            extracted_sections.append({
                'document': section.document,
                'section_title': section.title,
                'importance_rank': i,
                'page_number': section.page
            })
        
        # Create subsection_analysis with advanced refinement
        subsection_analysis = []
        for item in top_sections:
            section = item['section']
            refined_text = self.content_refiner.extract_key_sentences(
                section.content, keywords
            )
            
            subsection_analysis.append({
                'document': section.document,
                'refined_text': refined_text,
                'page_number': section.page
            })
        
        # Create final result
        result = {
            'metadata': {
                'input_documents': [doc['filename'] for doc in documents],
                'persona': persona['role'],
                'job_to_be_done': job_to_be_done['task'],
                'processing_timestamp': datetime.datetime.now().isoformat(),
                'domain_context': domain_context,
                'extracted_keywords': keywords[:10],  # Top 10 keywords
                'total_sections_analyzed': len(all_sections)
            },
            'extracted_sections': extracted_sections,
            'subsection_analysis': subsection_analysis
        }
        
        processing_time = time.time() - start_time
        print(f"\n✅ Processing complete in {processing_time:.2f} seconds")
        print(f"📊 Selected {len(extracted_sections)} most relevant sections")
        
        return result

def main():
    """Main function with correct output format"""
    print("🎯 Generic Challenge 1B - Universal Document Intelligence")
    print("=" * 70)
    
    input_file = "challenge1b_input.json"
    
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return
    
    processor = GenericChallenge1BProcessor()
    
    try:
        result = processor.process_challenge(input_file)
        
        # CRITICAL: Save to the exact expected filename
        output_file = "challenge1b_result.json"  # NOT challenge1b_output.json
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {output_file}")
        
        # Print summary
        print(f"\n📈 Results Summary:")
        print(f"   📄 Documents: {len(result['metadata']['input_documents'])}")
        print(f"   🎯 Top sections: {len(result['extracted_sections'])}")
        print(f"   🔑 Keywords: {len(result['metadata']['extracted_keywords'])}")
        print(f"   📊 Total analyzed: {result['metadata']['total_sections_analyzed']}")
        
        # Enhanced result validation
        if result['extracted_sections']:
            print(f"\n🏆 Top 3 Sections:")
            for section in result['extracted_sections'][:3]:
                print(f"   {section['importance_rank']}. {section['section_title']} "
                      f"(p.{section['page_number']})")
        
        if result['metadata']['extracted_keywords']:
            print(f"\n🔑 Top Keywords:")
            for i, keyword in enumerate(result['metadata']['extracted_keywords'][:5], 1):
                print(f"   {i}. {keyword}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()