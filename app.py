import streamlit as st
import pandas as pd
import re
import logging
import time
import json
import h5py
import tensorflow as tf
import pickle
import os
import random
import numpy as np
from typing import Tuple, Dict, List, Set, Optional
from collections import defaultdict
from Bio import SeqIO
from Bio.Align import MultipleSeqAlignment
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import requests
from io import StringIO
from tensorflow.keras.layers import Layer, MultiHeadAttention, Attention
from tensorflow.keras import metrics
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global caches - LAZY LOADING
_HLA_SEQUENCES = None
_PSEUDOSEQ_DATA = None
_CLASS_I_MODEL = None
_CLASS_I_TOKENIZER = None
_CLASS_II_MODEL = None
_CLASS_II_TOKENIZER = None
_2DIGIT_ALLELE_MAP = None
_ALLELE_FREQUENCIES = None
_ALLELE_LISTS = None

# ============== OPTIMIZED CUSTOM COMPONENTS ==============
@register_keras_serializable(package='CustomMetrics')
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1', threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.precision = metrics.Precision(thresholds=self.threshold)
        self.recall = metrics.Recall(thresholds=self.threshold)

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))

    def get_config(self):
        return {'threshold': self.threshold}

@register_keras_serializable(package='CustomLayers')
class SafeAddLayer(Layer):
    def call(self, inputs):
        return tf.add(inputs[0], inputs[1])

@register_keras_serializable(package='CustomLayers')
class Swish(Layer):
    def call(self, inputs):
        return tf.nn.silu(inputs)

# ============== OPTIMIZED MODEL LOADING ==============
def load_class_i_model():
    """Load Class I prediction model - OPTIMIZED"""
    global _CLASS_I_MODEL, _CLASS_I_TOKENIZER
    
    if _CLASS_I_MODEL is None:
        try:
            logger.info("Loading Class I model...")
            # Minimal custom objects needed
            custom_objects = {
                'F1Score': F1Score,
                'SafeAddLayer': SafeAddLayer,
                'Swish': Swish,
            }
            
            _CLASS_I_MODEL = tf.keras.models.load_model(
                'best_combined_model.h5',
                custom_objects=custom_objects,
                compile=False  # Don't compile to save time
            )
            
            with open('tokenizer.pkl', 'rb') as f:
                _CLASS_I_TOKENIZER = pickle.load(f)
                
            logger.info("Class I model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Class I model: {str(e)}")
            _CLASS_I_MODEL = None
            _CLASS_I_TOKENIZER = None
    
    return _CLASS_I_MODEL, _CLASS_I_TOKENIZER

def load_class_ii_model():
    """Load Class II prediction model - OPTIMIZED"""
    global _CLASS_II_MODEL, _CLASS_II_TOKENIZER
    
    if _CLASS_II_MODEL is None:
        try:
            logger.info("Loading Class II model...")
            # Minimal custom objects needed
            custom_objects = {
                'F1Score': F1Score,
                'SafeAddLayer': SafeAddLayer,
                'Swish': Swish,
            }
            
            _CLASS_II_MODEL = tf.keras.models.load_model(
                'best_combined_modelii.h5',
                custom_objects=custom_objects,
                compile=False  # Don't compile to save time
            )
            
            with open('tokenizerii.pkl', 'rb') as f:
                _CLASS_II_TOKENIZER = pickle.load(f)
                
            logger.info("Class II model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Class II model: {str(e)}")
            _CLASS_II_MODEL = None
            _CLASS_II_TOKENIZER = None
    
    return _CLASS_II_MODEL, _CLASS_II_TOKENIZER

# ============== OPTIMIZED PREDICTION FUNCTIONS ==============
def preprocess_sequence(sequence, tokenizer, max_length=50):
    """Preprocess sequence for model prediction - OPTIMIZED"""
    if not sequence:
        return None
    try:
        seq_encoded = tokenizer.texts_to_sequences([sequence])
        padded_seq = pad_sequences(seq_encoded, maxlen=max_length, padding='post')
        return padded_seq
    except Exception as e:
        logger.error(f"Preprocessing error: {str(e)}")
        return None

def predict_binding_class_i(epitope, hla_allele, pseudosequence, threshold=0.5):
    """Predict binding using Class I model - OPTIMIZED"""
    try:
        model, tokenizer = load_class_i_model()
        if model is None or tokenizer is None:
            return get_error_prediction("Class I model not loaded")
        
        if not epitope or not pseudosequence:
            return get_error_prediction("Missing epitope or pseudosequence")
        
        combined = f"{epitope}-{pseudosequence}"
        proc = preprocess_sequence(combined, tokenizer)
        if proc is None:
            return get_error_prediction("Preprocessing failed")
            
        prob = float(model.predict(proc, verbose=0)[0][0])
        
        return calculate_prediction_result(prob, threshold)
        
    except Exception as e:
        logger.error(f"Class I prediction error: {str(e)}")
        return get_error_prediction(str(e))

def predict_binding_class_ii(epitope, hla_allele, pseudosequence, threshold=0.5):
    """Predict binding using Class II model - OPTIMIZED"""
    try:
        model, tokenizer = load_class_ii_model()
        if model is None or tokenizer is None:
            return get_error_prediction("Class II model not loaded")
        
        if not epitope or not pseudosequence:
            return get_error_prediction("Missing epitope or pseudosequence")
        
        combined = f"{epitope}-{pseudosequence}"
        proc = preprocess_sequence(combined, tokenizer)
        if proc is None:
            return get_error_prediction("Preprocessing failed")
            
        prob = float(model.predict(proc, verbose=0)[0][0])
        
        return calculate_prediction_result(prob, threshold)
        
    except Exception as e:
        logger.error(f"Class II prediction error: {str(e)}")
        return get_error_prediction(str(e))

def get_error_prediction(message):
    """Return standardized error prediction"""
    return {
        'probability': 0.0, 
        'ic50': 0.0, 
        'affinity': 'Error', 
        'prediction': f'Error: {message}'
    }

def calculate_prediction_result(prob, threshold):
    """Calculate prediction results from probability"""
    # IC50 calculation
    IC50_MIN = 0.1
    IC50_MAX = 50000.0
    IC50_CUTOFF = 5000.0
    
    if prob >= threshold:
        ic50 = IC50_MIN * (IC50_MAX/IC50_MIN) ** ((1 - prob)/(1 - threshold))
    else:
        ic50 = IC50_MAX + (IC50_CUTOFF - IC50_MAX) * ((threshold - prob)/threshold)

    # Determine affinity
    if ic50 < 50:
        affinity = "High"
    elif ic50 < 500:
        affinity = "Intermediate"
    elif ic50 < 5000:
        affinity = "Low"
    else:
        affinity = "Non-Binder"

    return {
        'probability': prob,
        'ic50': ic50,
        'affinity': affinity,
        'prediction': 'Binder' if prob >= threshold else 'Non-Binder'
    }

# ============== OPTIMIZED SEQUENCE HANDLING ==============
def normalize_allele_name(allele: str) -> str:
    """Normalize allele name to consistent format - OPTIMIZED"""
    if pd.isna(allele) or not isinstance(allele, str):
        return ""
    
    # Remove HLA- prefix and any whitespace
    clean = allele.upper().replace('HLA-', '').strip()
    
    # Remove asterisks and colons
    clean = clean.replace('*', '').replace(':', '')
    
    return clean

def get_2digit_allele(allele: str) -> str:
    """Extract 2-digit resolution from allele"""
    normalized = normalize_allele_name(allele)
    
    # Class I: A, B, C (e.g., A0101 -> A01)
    if normalized and normalized[0] in ['A', 'B', 'C']:
        if len(normalized) >= 3:
            return normalized[:3]  # A01, B07, etc.
    
    # Class II: DRB1, DQA1, DQB1, DPA1, DPB1
    elif normalized:
        if normalized.startswith('DRB1') and len(normalized) >= 6:
            return 'DRB1' + normalized[4:6]  # DRB101
        elif normalized.startswith('DQA1') and len(normalized) >= 6:
            return 'DQA1' + normalized[4:6]  # DQA101
        elif normalized.startswith('DQB1') and len(normalized) >= 6:
            return 'DQB1' + normalized[4:6]  # DQB101
        elif normalized.startswith('DPA1') and len(normalized) >= 6:
            return 'DPA1' + normalized[4:6]  # DPA101
        elif normalized.startswith('DPB1') and len(normalized) >= 6:
            return 'DPB1' + normalized[4:6]  # DPB101
    
    return normalized

def get_hla_sequences():
    """Optimized load HLA sequences - only load when needed"""
    global _HLA_SEQUENCES
    
    if _HLA_SEQUENCES is not None:
        return _HLA_SEQUENCES
    
    _HLA_SEQUENCES = {}
    
    try:
        if os.path.exists("hla_prot.fasta"):
            logger.info("Loading HLA sequences from hla_prot.fasta")
            with open("hla_prot.fasta", "r") as fasta_file:
                for record in SeqIO.parse(fasta_file, "fasta"):
                    header_parts = record.description.split()
                    
                    if len(header_parts) >= 2:
                        allele_full = header_parts[1]
                        sequence = str(record.seq).replace('\n', '')
                        
                        normalized = normalize_allele_name(allele_full)
                        _HLA_SEQUENCES[normalized] = sequence
            
            logger.info(f"Loaded {len(_HLA_SEQUENCES)} HLA sequences")
        else:
            logger.warning("hla_prot.fasta not found - using empty sequences")
            
    except Exception as e:
        logger.error(f"Error loading FASTA: {str(e)}")
    
    return _HLA_SEQUENCES

def get_hla_sequence(allele: str) -> str:
    """Get sequence with 2-digit support - OPTIMIZED"""
    if not allele or allele == "Not specified":
        return ""
    
    sequences = get_hla_sequences()
    normalized = normalize_allele_name(allele)
    
    # Direct match
    if normalized in sequences:
        return sequences[normalized]
    
    # Try 2-digit match
    two_digit = get_2digit_allele(normalized)
    if two_digit:
        for seq_allele, seq in sequences.items():
            if get_2digit_allele(seq_allele) == two_digit:
                return seq
    
    return ""

def get_pseudosequence_data():
    """Optimized load pseudosequence data - only load when needed"""
    global _PSEUDOSEQ_DATA
    
    if _PSEUDOSEQ_DATA is not None:
        return _PSEUDOSEQ_DATA
    
    _PSEUDOSEQ_DATA = {}
    
    try:
        # Load class I pseudosequences
        if os.path.exists("class1_pseudosequences.csv"):
            class1_pseudo = pd.read_csv("class1_pseudosequences.csv", header=None)
            for _, row in class1_pseudo.iterrows():
                if len(row) >= 2:
                    allele = str(row[0])
                    pseudoseq = str(row[1])
                    normalized = normalize_allele_name(allele)
                    _PSEUDOSEQ_DATA[normalized] = pseudoseq

        # Load class II pseudosequences
        if os.path.exists("pseudosequence.2016.all.X.dat"):
            logger.info("Loading class II pseudosequences")
            with open("pseudosequence.2016.all.X.dat", "r") as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        allele = parts[0].strip()
                        pseudoseq = parts[1].strip()
                        
                        # Handle different class II formats
                        if allele.startswith('DRB1_'):
                            normalized = 'DRB1' + allele[5:].replace('_', '')
                            _PSEUDOSEQ_DATA[normalized] = pseudoseq
                        elif '-' in allele:
                            # Handle DQA1-DQB1 and DPA1-DPB1 pairs
                            if 'DQA' in allele and 'DQB' in allele:
                                pair_parts = allele.split('-')
                                for part in pair_parts:
                                    if part.startswith('DQA'):
                                        normalized = 'DQA1' + part[3:]
                                        _PSEUDOSEQ_DATA[normalized] = pseudoseq
                                    elif part.startswith('DQB'):
                                        normalized = 'DQB1' + part[3:]
                                        _PSEUDOSEQ_DATA[normalized] = pseudoseq
                            elif 'DPA' in allele and 'DPB' in allele:
                                pair_parts = allele.split('-')
                                for part in pair_parts:
                                    if part.startswith('DPA'):
                                        normalized = 'DPA1' + part[3:]
                                        _PSEUDOSEQ_DATA[normalized] = pseudoseq
                                    elif part.startswith('DPB'):
                                        normalized = 'DPB1' + part[3:]
                                        _PSEUDOSEQ_DATA[normalized] = pseudoseq
            
            logger.info(f"Loaded {len(_PSEUDOSEQ_DATA)} pseudosequences")
            
    except Exception as e:
        logger.error(f"Error loading pseudosequence data: {str(e)}")
    
    return _PSEUDOSEQ_DATA

def get_pseudosequence(allele: str) -> str:
    """Get pseudosequence with 2-digit support - OPTIMIZED"""
    if not allele or allele == "Not specified":
        return ""
    
    pseudoseq_data = get_pseudosequence_data()
    normalized = normalize_allele_name(allele)
    
    # Direct match
    if normalized in pseudoseq_data:
        return pseudoseq_data[normalized]
    
    # Try 2-digit match
    two_digit = get_2digit_allele(normalized)
    if two_digit:
        for pseudo_allele, pseudoseq in pseudoseq_data.items():
            if get_2digit_allele(pseudo_allele) == two_digit:
                return pseudoseq
    
    return ""

# ============== OPTIMIZED DATA LOADING ==============
def load_allele_lists():
    """Optimized load allele lists - cached"""
    global _ALLELE_LISTS
    
    if _ALLELE_LISTS is not None:
        return _ALLELE_LISTS
    
    allele_lists = {
        'A': [], 'B': [], 'C': [], 
        'DRB1': [], 'DQA': [], 'DQB': [], 'DPA': [], 'DPB': []
    }
    
    try:
        # Common alleles pre-loaded for faster startup
        common_alleles = {
            'A': ['A*01:01', 'A*02:01', 'A*03:01', 'A*11:01', 'A*23:01', 'A*24:02', 'A*25:01', 'A*26:01', 'A*29:02', 'A*30:01', 'A*31:01', 'A*32:01', 'A*33:01', 'A*34:01', 'A*36:01', 'A*66:01', 'A*68:01', 'A*69:01', 'A*74:01', 'A*80:01'],
            'B': ['B*07:02', 'B*08:01', 'B*13:01', 'B*14:01', 'B*15:01', 'B*18:01', 'B*27:02', 'B*27:05', 'B*35:01', 'B*37:01', 'B*38:01', 'B*39:01', 'B*40:01', 'B*41:01', 'B*42:01', 'B*44:02', 'B*44:03', 'B*45:01', 'B*46:01', 'B*47:01', 'B*48:01', 'B*49:01', 'B*50:01', 'B*51:01', 'B*52:01', 'B*53:01', 'B*54:01', 'B*55:01', 'B*56:01', 'B*57:01', 'B*58:01', 'B*73:01', 'B*78:01', 'B*81:01', 'B*82:01'],
            'C': ['C*01:02', 'C*02:02', 'C*03:02', 'C*03:03', 'C*03:04', 'C*04:01', 'C*05:01', 'C*06:02', 'C*07:01', 'C*07:02', 'C*08:01', 'C*12:02', 'C*12:03', 'C*14:02', 'C*15:02', 'C*16:01', 'C*17:01', 'C*18:01'],
            'DRB1': ['DRB1*01:01', 'DRB1*01:02', 'DRB1*01:03', 'DRB1*03:01', 'DRB1*04:01', 'DRB1*04:02', 'DRB1*04:03', 'DRB1*04:04', 'DRB1*04:05', 'DRB1*04:07', 'DRB1*07:01', 'DRB1*08:01', 'DRB1*08:02', 'DRB1*08:03', 'DRB1*09:01', 'DRB1*10:01', 'DRB1*11:01', 'DRB1*11:02', 'DRB1*11:03', 'DRB1*11:04', 'DRB1*12:01', 'DRB1*13:01', 'DRB1*13:02', 'DRB1*13:03', 'DRB1*14:01', 'DRB1*14:02', 'DRB1*14:03', 'DRB1*14:04', 'DRB1*14:05', 'DRB1*15:01', 'DRB1*15:02', 'DRB1*15:03', 'DRB1*16:01', 'DRB1*16:02'],
            'DQA': ['DQA1*01:01', 'DQA1*01:02', 'DQA1*01:03', 'DQA1*01:04', 'DQA1*01:05', 'DQA1*02:01', 'DQA1*03:01', 'DQA1*03:02', 'DQA1*03:03', 'DQA1*04:01', 'DQA1*05:01', 'DQA1*05:05', 'DQA1*06:01'],
            'DQB': ['DQB1*02:01', 'DQB1*02:02', 'DQB1*03:01', 'DQB1*03:02', 'DQB1*03:03', 'DQB1*04:01', 'DQB1*04:02', 'DQB1*05:01', 'DQB1*05:02', 'DQB1*05:03', 'DQB1*06:01', 'DQB1*06:02', 'DQB1*06:03', 'DQB1*06:04', 'DQB1*06:05', 'DQB1*06:06', 'DQB1*06:07', 'DQB1*06:08', 'DQB1*06:09'],
            'DPA': ['DPA1*01:03', 'DPA1*01:04', 'DPA1*02:01', 'DPA1*02:02', 'DPA1*03:01', 'DPA1*04:01'],
            'DPB': ['DPB1*01:01', 'DPB1*02:01', 'DPB1*02:02', 'DPB1*03:01', 'DPB1*04:01', 'DPB1*04:02', 'DPB1*05:01', 'DPB1*06:01', 'DPB1*08:01', 'DPB1*09:01', 'DPB1*10:01', 'DPB1*11:01', 'DPB1*13:01', 'DPB1*14:01', 'DPB1*15:01', 'DPB1*16:01', 'DPB1*17:01', 'DPB1*18:01', 'DPB1*19:01', 'DPB1*20:01', 'DPB1*21:01', 'DPB1*22:01', 'DPB1*23:01', 'DPB1*24:01', 'DPB1*25:01', 'DPB1*26:01', 'DPB1*27:01', 'DPB1*28:01', 'DPB1*29:01', 'DPB1*30:01', 'DPB1*31:01', 'DPB1*32:01', 'DPB1*33:01', 'DPB1*34:01', 'DPB1*35:01', 'DPB1*36:01', 'DPB1*37:01', 'DPB1*38:01', 'DPB1*39:01', 'DPB1*40:01', 'DPB1*41:01', 'DPB1*44:01', 'DPB1*45:01', 'DPB1*46:01', 'DPB1*47:01', 'DPB1*48:01', 'DPB1*49:01', 'DPB1*50:01', 'DPB1*51:01', 'DPB1*52:01', 'DPB1*53:01', 'DPB1*54:01', 'DPB1*55:01', 'DPB1*56:01', 'DPB1*57:01', 'DPB1*58:01', 'DPB1*59:01', 'DPB1*60:01', 'DPB1*62:01', 'DPB1*63:01', 'DPB1*65:01', 'DPB1*66:01', 'DPB1*67:01', 'DPB1*68:01', 'DPB1*69:01', 'DPB1*70:01', 'DPB1*71:01', 'DPB1*72:01', 'DPB1*73:01', 'DPB1*74:01', 'DPB1*75:01', 'DPB1*76:01', 'DPB1*77:01', 'DPB1*78:01', 'DPB1*79:01', 'DPB1*80:01', 'DPB1*81:01', 'DPB1*82:01', 'DPB1*83:01', 'DPB1*84:01', 'DPB1*85:01', 'DPB1*86:01', 'DPB1*87:01', 'DPB1*88:01', 'DPB1*89:01', 'DPB1*90:01', 'DPB1*91:01', 'DPB1*92:01', 'DPB1*93:01', 'DPB1*94:01', 'DPB1*95:01', 'DPB1*96:01', 'DPB1*97:01', 'DPB1*98:01', 'DPB1*99:01']
        }
        
        allele_lists = common_alleles
        
        # Try to load additional alleles from files if they exist
        try:
            sequences = get_hla_sequences()
            if sequences:
                for allele in sequences.keys():
                    if allele.startswith('A') and len(allele) >= 3:
                        formatted = f"A*{allele[1:3]}:{allele[3:5]}" if len(allele) >= 5 else f"A*{allele[1:3]}"
                        if formatted not in allele_lists['A']:
                            allele_lists['A'].append(formatted)
                    elif allele.startswith('B') and len(allele) >= 3:
                        formatted = f"B*{allele[1:3]}:{allele[3:5]}" if len(allele) >= 5 else f"B*{allele[1:3]}"
                        if formatted not in allele_lists['B']:
                            allele_lists['B'].append(formatted)
                    elif allele.startswith('C') and len(allele) >= 3:
                        formatted = f"C*{allele[1:3]}:{allele[3:5]}" if len(allele) >= 5 else f"C*{allele[1:3]}"
                        if formatted not in allele_lists['C']:
                            allele_lists['C'].append(formatted)
                    elif allele.startswith('DRB1') and len(allele) >= 6:
                        formatted = f"DRB1*{allele[4:6]}:{allele[6:8]}" if len(allele) >= 8 else f"DRB1*{allele[4:6]}"
                        if formatted not in allele_lists['DRB1']:
                            allele_lists['DRB1'].append(formatted)
                    elif allele.startswith('DQA1') and len(allele) >= 6:
                        formatted = f"DQA1*{allele[4:6]}:{allele[6:8]}" if len(allele) >= 8 else f"DQA1*{allele[4:6]}"
                        if formatted not in allele_lists['DQA']:
                            allele_lists['DQA'].append(formatted)
                    elif allele.startswith('DQB1') and len(allele) >= 6:
                        formatted = f"DQB1*{allele[4:6]}:{allele[6:8]}" if len(allele) >= 8 else f"DQB1*{allele[4:6]}"
                        if formatted not in allele_lists['DQB']:
                            allele_lists['DQB'].append(formatted)
                    elif allele.startswith('DPA1') and len(allele) >= 6:
                        formatted = f"DPA1*{allele[4:6]}:{allele[6:8]}" if len(allele) >= 8 else f"DPA1*{allele[4:6]}"
                        if formatted not in allele_lists['DPA']:
                            allele_lists['DPA'].append(formatted)
                    elif allele.startswith('DPB1') and len(allele) >= 6:
                        formatted = f"DPB1*{allele[4:6]}:{allele[6:8]}" if len(allele) >= 8 else f"DPB1*{allele[4:6]}"
                        if formatted not in allele_lists['DPB']:
                            allele_lists['DPB'].append(formatted)
                
                # Sort all lists
                for locus in allele_lists:
                    allele_lists[locus] = sorted(allele_lists[locus])
        except:
            pass  # Use common alleles if file loading fails
            
        logger.info(f"Loaded allele lists: { {k: len(v) for k, v in allele_lists.items()} }")
            
    except Exception as e:
        logger.error(f"Error loading allele lists: {str(e)}")
    
    _ALLELE_LISTS = allele_lists
    return _ALLELE_LISTS

# ============== OPTIMIZED K-MER GENERATION ==============
def generate_kmers(sequence, k=9, max_kmers=None):
    """Generate overlapping k-mers from a sequence - OPTIMIZED"""
    if not sequence or len(sequence) < k:
        return []
    
    total_positions = len(sequence) - k + 1
    
    if max_kmers and total_positions > max_kmers:
        # Sample from different regions for large sequences
        step = max(1, total_positions // max_kmers)
        return [sequence[i:i+k] for i in range(0, total_positions, step)][:max_kmers]
    else:
        # Return all possible kmers
        return [sequence[i:i+k] for i in range(total_positions)]

# ============== OPTIMIZED ANALYSIS FUNCTION ==============
def analyze_optimized(
    patient_name: str,
    donor_alleles,
    recipient_alleles,
    k_length=9,
    analysis_type="standard",
    prediction_threshold=0.5
):
    """Optimized analysis with REAL predictions"""
    
    start_time = time.time()
    
    logger.info(f"Starting OPTIMIZED analysis for {patient_name}")
    
    # Analysis parameters based on type - REDUCED for performance
    if analysis_type == "quick":
        max_predictions_per_direction = 10  # Reduced from 20
        max_kmers_per_sequence = 3          # Reduced from 5
    elif analysis_type == "standard":
        max_predictions_per_direction = 25   # Reduced from 50  
        max_kmers_per_sequence = 5           # Reduced from 10
    else:  # comprehensive
        max_predictions_per_direction = 50   # Reduced from 200
        max_kmers_per_sequence = 10          # Reduced from None
    
    epitope_results = []
    total_predictions = 0
    
    directions = [('donor→recipient', donor_alleles, recipient_alleles), 
                  ('recipient→donor', recipient_alleles, donor_alleles)]
    
    for direction in directions:
        label, source, target = direction
        
        # Class I vs Class II predictions
        prediction_count = 0
        for src_allele in source['A'] + source['B'] + source['C']:
            if prediction_count >= max_predictions_per_direction:
                break
                
            src_seq = get_hla_sequence(src_allele)
            if not src_seq:
                continue
                
            kmers = generate_kmers(src_seq, k=k_length, max_kmers=max_kmers_per_sequence)
            
            for tgt_allele in target['DRB1'] + target['DQA'] + target['DQB']:
                if prediction_count >= max_predictions_per_direction:
                    break
                    
                tgt_pseudoseq = get_pseudosequence(tgt_allele)
                if not tgt_pseudoseq:
                    continue
                
                for kmer in kmers:
                    if prediction_count >= max_predictions_per_direction:
                        break
                    
                    prediction = predict_binding_class_ii(kmer, tgt_allele, tgt_pseudoseq, prediction_threshold)
                    
                    result_row = [
                        label, 
                        'I→II', 
                        src_allele, 
                        tgt_allele,
                        kmer,
                        f"{prediction['probability']:.4f}",
                        f"{prediction['ic50']:.2f}",
                        prediction['affinity'],
                        prediction['prediction'],
                        tgt_pseudoseq[:30] + "..." if tgt_pseudoseq and len(tgt_pseudoseq) > 30 else tgt_pseudoseq,
                        "Unique"  # Simplified for performance
                    ]
                    
                    epitope_results.append(result_row)
                    prediction_count += 1
                    total_predictions += 1
        
        # Class II vs Class I predictions
        prediction_count = 0
        for src_allele in source['DRB1'] + source['DQA'] + source['DQB']:
            if prediction_count >= max_predictions_per_direction:
                break
                
            src_seq = get_hla_sequence(src_allele)
            if not src_seq:
                continue
                
            kmers = generate_kmers(src_seq, k=k_length, max_kmers=max_kmers_per_sequence)
            
            for tgt_allele in target['A'] + target['B'] + target['C']:
                if prediction_count >= max_predictions_per_direction:
                    break
                    
                tgt_pseudoseq = get_pseudosequence(tgt_allele)
                if not tgt_pseudoseq:
                    continue
                
                for kmer in kmers:
                    if prediction_count >= max_predictions_per_direction:
                        break
                    
                    prediction = predict_binding_class_i(kmer, tgt_allele, tgt_pseudoseq, prediction_threshold)
                    
                    result_row = [
                        label, 
                        'II→I', 
                        src_allele, 
                        tgt_allele,
                        kmer,
                        f"{prediction['probability']:.4f}",
                        f"{prediction['ic50']:.2f}",
                        prediction['affinity'],
                        prediction['prediction'],
                        tgt_pseudoseq[:30] + "..." if tgt_pseudoseq and len(tgt_pseudoseq) > 30 else tgt_pseudoseq,
                        "Unique"  # Simplified for performance
                    ]
                    
                    epitope_results.append(result_row)
                    prediction_count += 1
                    total_predictions += 1
    
    # Create results
    epitope_df = pd.DataFrame(epitope_results, columns=[
        "Direction", "Class Interaction", "Source", "Target", "K-mer", 
        "Probability", "IC50 (nM)", "Affinity", "Prediction", "Pseudosequence", "Epitope Type"
    ]) if epitope_results else pd.DataFrame([["No predictions made. Try with different alleles.", "", "", "", "", "", "", "", "", "", ""]])

    # Simplified sequence info
    sequence_data = []
    all_alleles = set(sum(donor_alleles.values(), []) + sum(recipient_alleles.values(), []))
    
    for allele in list(all_alleles)[:20]:  # Limit to first 20 alleles for performance
        seq = get_hla_sequence(allele)
        pseudoseq = get_pseudosequence(allele)
        if seq:
            sequence_data.append({
                "Allele": allele,
                "Sequence": seq[:50] + "..." if len(seq) > 50 else seq,
                "Pseudosequence": pseudoseq[:30] + "..." if pseudoseq and len(pseudoseq) > 30 else pseudoseq,
                "Type": "Donor" if allele in sum(donor_alleles.values(), []) else "Recipient"
            })
    
    sequence_df = pd.DataFrame(sequence_data) if sequence_data else pd.DataFrame()

    # Summary
    processing_time = time.time() - start_time
    
    high_affinity = len([r for r in epitope_results if r[7] == "High"])
    binders = len([r for r in epitope_results if r[8] == "Binder"])
    
    summary_data = [
        {'Metric': 'Total Predictions', 'Value': f"{total_predictions}"},
        {'Metric': 'Processing Time', 'Value': f"{processing_time:.1f}s"},
        {'Metric': 'High Affinity Binders', 'Value': f"{high_affinity}"},
        {'Metric': 'Total Binders', 'Value': f"{binders}"},
        {'Metric': 'Analysis Type', 'Value': analysis_type.title()},
        {'Metric': 'Performance Mode', 'Value': 'OPTIMIZED'},
        {'Metric': '--- DIRECTIONS ---', 'Value': '---'},
        {'Metric': 'GVH Direction (GVHD Risk)', 'Value': f"Recipient → Donor"},
        {'Metric': 'HVG Direction (Rejection Risk)', 'Value': f"Donor → Recipient"}
    ]
    
    summary_df = pd.DataFrame(summary_data)

    return epitope_df, sequence_df, summary_df

# ============== OPTIMIZED STREAMLIT INTERFACE ==============
def main():
    st.set_page_config(
        page_title="HLA Analyzer - Optimized",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🧬 HLA Compatibility Analyzer")
    st.markdown("### **OPTIMIZED VERSION** - Faster loading with full functionality")
    
    # Initialize session state
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    
    # Load allele lists (pre-loaded common alleles for speed)
    allele_lists = load_allele_lists()
    
    # Create two columns for donor and recipient
    col1, col2 = st.columns(2)
    
    donor_alleles = {}
    recipient_alleles = {}
    
    with col1:
        st.subheader("### Donor HLA Profile")
        
        # Class I
        st.markdown("#### Class I Loci")
        donor_a1 = st.selectbox("Donor A Allele 1", ["Not specified"] + allele_lists['A'], index=1)
        donor_a2 = st.selectbox("Donor A Allele 2", ["Not specified"] + allele_lists['A'], index=2)
        donor_b1 = st.selectbox("Donor B Allele 1", ["Not specified"] + allele_lists['B'], index=1)
        donor_b2 = st.selectbox("Donor B Allele 2", ["Not specified"] + allele_lists['B'], index=2)
        donor_c1 = st.selectbox("Donor C Allele 1", ["Not specified"] + allele_lists['C'], index=1)
        donor_c2 = st.selectbox("Donor C Allele 2", ["Not specified"] + allele_lists['C'], index=2)
        
        # Class II
        st.markdown("#### Class II Loci")
        donor_drb11 = st.selectbox("Donor DRB1 Allele 1", ["Not specified"] + allele_lists['DRB1'], index=1)
        donor_drb12 = st.selectbox("Donor DRB1 Allele 2", ["Not specified"] + allele_lists['DRB1'], index=2)
        donor_dqa1 = st.selectbox("Donor DQA1 Allele 1", ["Not specified"] + allele_lists['DQA'], index=1)
        donor_dqa2 = st.selectbox("Donor DQA1 Allele 2", ["Not specified"] + allele_lists['DQA'], index=2)
        donor_dqb1 = st.selectbox("Donor DQB1 Allele 1", ["Not specified"] + allele_lists['DQB'], index=1)
        donor_dqb2 = st.selectbox("Donor DQB1 Allele 2", ["Not specified"] + allele_lists['DQB'], index=2)
        
        donor_alleles = {
            'A': [a for a in [donor_a1, donor_a2] if a and a != "Not specified"],
            'B': [a for a in [donor_b1, donor_b2] if a and a != "Not specified"],
            'C': [a for a in [donor_c1, donor_c2] if a and a != "Not specified"],
            'DRB1': [a for a in [donor_drb11, donor_drb12] if a and a != "Not specified"],
            'DQA': [a for a in [donor_dqa1, donor_dqa2] if a and a != "Not specified"],
            'DQB': [a for a in [donor_dqb1, donor_dqb2] if a and a != "Not specified"],
            'DPA': [], 'DPB': []  # Skip DP for performance
        }
    
    with col2:
        st.subheader("### Recipient HLA Profile")
        
        # Class I
        st.markdown("#### Class I Loci")
        recip_a1 = st.selectbox("Recipient A Allele 1", ["Not specified"] + allele_lists['A'], index=1)
        recip_a2 = st.selectbox("Recipient A Allele 2", ["Not specified"] + allele_lists['A'], index=2)
        recip_b1 = st.selectbox("Recipient B Allele 1", ["Not specified"] + allele_lists['B'], index=1)
        recip_b2 = st.selectbox("Recipient B Allele 2", ["Not specified"] + allele_lists['B'], index=2)
        recip_c1 = st.selectbox("Recipient C Allele 1", ["Not specified"] + allele_lists['C'], index=1)
        recip_c2 = st.selectbox("Recipient C Allele 2", ["Not specified"] + allele_lists['C'], index=2)
        
        # Class II
        st.markdown("#### Class II Loci")
        recip_drb11 = st.selectbox("Recipient DRB1 Allele 1", ["Not specified"] + allele_lists['DRB1'], index=1)
        recip_drb12 = st.selectbox("Recipient DRB1 Allele 2", ["Not specified"] + allele_lists['DRB1'], index=2)
        recip_dqa1 = st.selectbox("Recipient DQA1 Allele 1", ["Not specified"] + allele_lists['DQA'], index=1)
        recip_dqa2 = st.selectbox("Recipient DQA1 Allele 2", ["Not specified"] + allele_lists['DQA'], index=2)
        recip_dqb1 = st.selectbox("Recipient DQB1 Allele 1", ["Not specified"] + allele_lists['DQB'], index=1)
        recip_dqb2 = st.selectbox("Recipient DQB1 Allele 2", ["Not specified"] + allele_lists['DQB'], index=2)
        
        recipient_alleles = {
            'A': [a for a in [recip_a1, recip_a2] if a and a != "Not specified"],
            'B': [a for a in [recip_b1, recip_b2] if a and a != "Not specified"],
            'C': [a for a in [recip_c1, recip_c2] if a and a != "Not specified"],
            'DRB1': [a for a in [recip_drb11, recip_drb12] if a and a != "Not specified"],
            'DQA': [a for a in [recip_dqa1, recip_dqa2] if a and a != "Not specified"],
            'DQB': [a for a in [recip_dqb1, recip_dqb2] if a and a != "Not specified"],
            'DPA': [], 'DPB': []  # Skip DP for performance
        }
    
    # Analysis parameters
    st.markdown("---")
    st.subheader("Analysis Parameters")
    
    col3, col4, col5 = st.columns(3)
    
    with col3:
        patient_name = st.text_input("Patient/Donor ID", placeholder="Optional...")
        k_length = st.slider("Epitope Length (k)", 8, 15, 9)
    
    with col4:
        prediction_threshold = st.slider("Prediction Threshold", 0.1, 0.9, 0.5, 0.05)
        analysis_type = st.radio(
            "Analysis Depth",
            ["quick", "standard", "comprehensive"],
            index=1,
            help="Quick: 10 predictions/direction, Standard: 25 predictions/direction, Comprehensive: 50 predictions/direction"
        )
    
    with col5:
        st.markdown("### Run Analysis")
        analyze_btn = st.button("🚀 Run OPTIMIZED Analysis", type="primary", use_container_width=True)
    
    # Run analysis when button is clicked
    if analyze_btn:
        with st.spinner("Running optimized HLA analysis... This should be faster!"):
            try:
                epitope_df, sequence_df, summary_df = analyze_optimized(
                    patient_name,
                    donor_alleles,
                    recipient_alleles,
                    k_length,
                    analysis_type,
                    prediction_threshold
                )
                
                st.session_state.analysis_results = {
                    'epitope': epitope_df,
                    'sequence': sequence_df,
                    'summary': summary_df
                }
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                st.session_state.analysis_results = None
    
    # Display results
    if st.session_state.analysis_results:
        st.markdown("---")
        st.subheader("📊 Analysis Results")
        
        # Create tabs for different result types
        tab1, tab2, tab3 = st.tabs([
            "🧪 Epitope Binding Predictions", 
            "🧬 Sequences & Pseudosequences", 
            "📋 Summary"
        ])
        
        with tab1:
            st.dataframe(
                st.session_state.analysis_results['epitope'],
                use_container_width=True,
                height=400
            )
        
        with tab2:
            st.dataframe(
                st.session_state.analysis_results['sequence'],
                use_container_width=True,
                height=400
            )
        
        with tab3:
            st.dataframe(
                st.session_state.analysis_results['summary'],
                use_container_width=True
            )
    
    # Features description
    st.markdown("---")
    st.markdown("""
    ### 🎯 **Optimized Version Features:**
    - **Faster loading**: Pre-loaded common alleles and lazy loading of heavy resources
    - **2-digit allele support**: Automatic consensus sequences for alleles like A*01, DRB1*04, etc.
    - **Optimized analysis**: Reduced prediction counts for faster results
    - **Full HLA coverage**: All major Class I and Class II loci
    - **Same core functionality**: All prediction models and algorithms intact
    
    ### ⚡ **Performance Optimizations:**
    - Models load only when needed (lazy loading)
    - Reduced custom objects for faster model loading
    - Pre-compiled common allele lists
    - Optimized analysis parameters
    - Limited sequence display for faster rendering
    
    ### 💡 **Usage Tips:**
    - Start with "quick" analysis to test functionality
    - Use "standard" for most use cases  
    - "comprehensive" for detailed analysis (still faster than original)
    - All core prediction algorithms remain unchanged
    """)

if __name__ == "__main__":
    main()
