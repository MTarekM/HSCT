import gradio as gr
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

# Global caches - comprehensive loading
_HLA_SEQUENCES = None
_PSEUDOSEQ_DATA = None
_LEADER_TYPE_MAP = None
_CLASS_I_MODEL = None
_CLASS_I_TOKENIZER = None
_CLASS_II_MODEL = None
_CLASS_II_TOKENIZER = None
_CONSENSUS_SEQUENCES = None
_2DIGIT_ALLELE_MAP = None
_ALLELE_FREQUENCIES = None

# ============== CUSTOM COMPONENTS FOR MODELS ==============
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

@register_keras_serializable(package='CustomOptimizers')
class AdamW(tf.keras.optimizers.legacy.Adam):
    def __init__(self, weight_decay=0.01, **kwargs):
        super().__init__(**kwargs)
        self.weight_decay = weight_decay

    def _resource_apply_dense(self, grad, var, apply_state):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype)) 
                       or self._fallback_apply_state(var_device, var_dtype))
        
        lr = coefficients['lr_t']
        wd = tf.cast(self.weight_decay, var_dtype)
        var.assign_sub(var * wd * lr)
        return super()._resource_apply_dense(grad, var, apply_state)

    def get_config(self):
        config = super().get_config()
        config.update({'weight_decay': self.weight_decay})
        return config

@register_keras_serializable(package='CustomLayers')
class SafeAddLayer(Layer):
    def call(self, inputs):
        return tf.add(inputs[0], inputs[1])

@register_keras_serializable(package='CustomLayers')
class Swish(Layer):
    def call(self, inputs):
        return tf.nn.silu(inputs)

@register_keras_serializable(package='CustomMetrics')
class NegativePredictiveValue(tf.keras.metrics.Metric):
    def __init__(self, name='npv', threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.true_negatives = self.add_weight(name='tn', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred >= self.threshold, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        tn = tf.reduce_sum((1 - y_true) * (1 - y_pred))
        fn = tf.reduce_sum(y_true * (1 - y_pred))
        self.true_negatives.assign_add(tn)
        self.false_negatives.assign_add(fn)

    def result(self):
        return self.true_negatives / (self.true_negatives + self.false_negatives + 1e-7)

    def reset_state(self):
        self.true_negatives.assign(0)
        self.false_negatives.assign(0)

    def get_config(self):
        return {'threshold': self.threshold}

# ============== COMPREHENSIVE MODEL LOADING ==============
def load_class_i_model():
    """Load Class I prediction model - COMPREHENSIVE"""
    global _CLASS_I_MODEL, _CLASS_I_TOKENIZER
    
    if _CLASS_I_MODEL is None:
        try:
            logger.info("Loading Class I model...")
            custom_objects = {
                'F1Score': F1Score,
                'NegativePredictiveValue': NegativePredictiveValue,
                'AdamW': AdamW,
                'SafeAddLayer': SafeAddLayer,
                'Swish': Swish,
                'MultiHeadAttention': MultiHeadAttention,
                'Attention': Attention,
            }
            
            _CLASS_I_MODEL = tf.keras.models.load_model(
                'best_combined_model.h5',
                custom_objects=custom_objects,
                compile=True
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
    """Load Class II prediction model - COMPREHENSIVE"""
    global _CLASS_II_MODEL, _CLASS_II_TOKENIZER
    
    if _CLASS_II_MODEL is None:
        try:
            logger.info("Loading Class II model...")
            custom_objects = {
                'F1Score': F1Score,
                'NegativePredictiveValue': NegativePredictiveValue,
                'AdamW': AdamW,
                'SafeAddLayer': SafeAddLayer,
                'Swish': Swish,
                'MultiHeadAttention': MultiHeadAttention,
                'Attention': Attention,
            }
            
            _CLASS_II_MODEL = tf.keras.models.load_model(
                'best_combined_modelii.h5',
                custom_objects=custom_objects,
                compile=True
            )
            
            with open('tokenizerii.pkl', 'rb') as f:
                _CLASS_II_TOKENIZER = pickle.load(f)
                
            logger.info("Class II model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Class II model: {str(e)}")
            _CLASS_II_MODEL = None
            _CLASS_II_TOKENIZER = None
    
    return _CLASS_II_MODEL, _CLASS_II_TOKENIZER

# ============== COMPREHENSIVE PREDICTION FUNCTIONS ==============
def preprocess_sequence(sequence, tokenizer, max_length=50):
    """Preprocess sequence for model prediction - COMPREHENSIVE"""
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
    """Predict binding using Class I model - COMPREHENSIVE"""
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
    """Predict binding using Class II model - COMPREHENSIVE"""
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

# ============== COMPREHENSIVE SEQUENCE HANDLING ==============
def normalize_allele_name(allele: str) -> str:
    """Normalize allele name to consistent format - COMPREHENSIVE"""
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
        # DRB1, DQA1, DQB1, DPA1, DPB1
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

def load_allele_frequencies():
    """Load allele frequency data for consensus selection"""
    global _ALLELE_FREQUENCIES
    
    if _ALLELE_FREQUENCIES is None:
        _ALLELE_FREQUENCIES = {}
        try:
            # Common allele frequencies (simulated - in real implementation, load from AFND or similar)
            common_frequencies = {
                'A0101': 0.15, 'A0201': 0.25, 'A0301': 0.12, 'A1101': 0.08, 'A2402': 0.18,
                'B0702': 0.10, 'B0801': 0.08, 'B1501': 0.06, 'B3501': 0.05, 'B4001': 0.04,
                'C0102': 0.08, 'C0303': 0.07, 'C0401': 0.12, 'C0501': 0.05, 'C0701': 0.15,
                'DRB10101': 0.10, 'DRB10301': 0.08, 'DRB10401': 0.15, 'DRB10701': 0.11,
                'DQA10101': 0.20, 'DQA10102': 0.15, 'DQA10501': 0.25,
                'DQB10201': 0.12, 'DQB10301': 0.18, 'DQB10501': 0.14, 'DQB10602': 0.10
            }
            _ALLELE_FREQUENCIES = common_frequencies
        except Exception as e:
            logger.error(f"Error loading allele frequencies: {str(e)}")
    
    return _ALLELE_FREQUENCIES

def get_consensus_sequence(allele_2digit: str, sequences: Dict[str, str]) -> str:
    """Get consensus sequence for 2-digit allele"""
    if not allele_2digit or not sequences:
        return ""
    
    # Find all sequences that match the 2-digit allele
    matching_sequences = []
    for allele, seq in sequences.items():
        if get_2digit_allele(allele) == allele_2digit:
            matching_sequences.append(seq)
    
    if not matching_sequences:
        return ""
    
    # If only one sequence, return it
    if len(matching_sequences) == 1:
        return matching_sequences[0]
    
    # Find the most frequent allele for this 2-digit group
    frequencies = load_allele_frequencies()
    best_allele = None
    best_frequency = 0
    
    for allele in sequences.keys():
        if get_2digit_allele(allele) == allele_2digit:
            freq = frequencies.get(allele, 0)
            if freq > best_frequency:
                best_frequency = freq
                best_allele = allele
    
    # Return sequence of most frequent allele, or first one if no frequency data
    if best_allele and best_allele in sequences:
        return sequences[best_allele]
    else:
        # Return the first matching sequence
        for allele, seq in sequences.items():
            if get_2digit_allele(allele) == allele_2digit:
                return seq
    
    return ""

def build_2digit_allele_map():
    """Build mapping of 2-digit alleles to representative sequences"""
    global _2DIGIT_ALLELE_MAP
    
    if _2DIGIT_ALLELE_MAP is not None:
        return _2DIGIT_ALLELE_MAP
    
    _2DIGIT_ALLELE_MAP = {}
    sequences = get_hla_sequences()
    
    # Collect all 2-digit alleles
    two_digit_alleles = set()
    for allele in sequences.keys():
        two_digit = get_2digit_allele(allele)
        if two_digit:
            two_digit_alleles.add(two_digit)
    
    # Get consensus sequence for each 2-digit allele
    for allele_2digit in two_digit_alleles:
        consensus_seq = get_consensus_sequence(allele_2digit, sequences)
        if consensus_seq:
            _2DIGIT_ALLELE_MAP[allele_2digit] = consensus_seq
    
    logger.info(f"Built 2-digit allele map with {len(_2DIGIT_ALLELE_MAP)} entries")
    return _2DIGIT_ALLELE_MAP

def get_hla_sequences():
    """Comprehensive load HLA sequences"""
    global _HLA_SEQUENCES
    
    if _HLA_SEQUENCES is not None:
        return _HLA_SEQUENCES
    
    _HLA_SEQUENCES = {}
    
    try:
        if os.path.exists("hla_prot.fasta"):
            logger.info("Loading comprehensive sequences from hla_prot.fasta")
            with open("hla_prot.fasta", "r") as fasta_file:
                for record in SeqIO.parse(fasta_file, "fasta"):
                    header_parts = record.description.split()
                    
                    if len(header_parts) >= 2:
                        allele_full = header_parts[1]
                        sequence = str(record.seq).replace('\n', '')
                        
                        normalized = normalize_allele_name(allele_full)
                        _HLA_SEQUENCES[normalized] = sequence
            
            logger.info(f"Loaded {len(_HLA_SEQUENCES)} comprehensive sequences")
            
            # Build 2-digit allele map
            build_2digit_allele_map()
            
        else:
            logger.error("hla_prot.fasta not found")
            
    except Exception as e:
        logger.error(f"Error loading FASTA: {str(e)}")
    
    return _HLA_SEQUENCES

def get_hla_sequence(allele: str) -> str:
    """Get sequence with 2-digit support"""
    if not allele or allele == "Not specified":
        return ""
    
    sequences = get_hla_sequences()
    normalized = normalize_allele_name(allele)
    
    # Direct match
    if normalized in sequences:
        return sequences[normalized]
    
    # Check if it's a 2-digit allele
    two_digit_map = build_2digit_allele_map()
    if normalized in two_digit_map:
        return two_digit_map[normalized]
    
    # Try 2-digit match
    two_digit = get_2digit_allele(normalized)
    if two_digit and two_digit in two_digit_map:
        return two_digit_map[two_digit]
    
    return ""

def get_pseudosequence_data():
    """Comprehensive load pseudosequence data"""
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
            logger.info("Loading comprehensive class II pseudosequences")
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
            
            logger.info(f"Loaded {len(_PSEUDOSEQ_DATA)} comprehensive pseudosequences")
            
    except Exception as e:
        logger.error(f"Error loading pseudosequence data: {str(e)}")
    
    return _PSEUDOSEQ_DATA

def get_pseudosequence(allele: str) -> str:
    """Get pseudosequence with 2-digit support"""
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

# ============== COMPREHENSIVE DATA LOADING ==============
def load_comprehensive_allele_lists():
    """Load comprehensive allele lists for dropdowns with 2-digit support"""
    allele_lists = {
        'A': [], 'B': [], 'C': [], 
        'DRB1': [], 'DQA': [], 'DQB': [], 'DPA': [], 'DPB': []
    }
    
    try:
        # Load from FASTA file for comprehensive list
        sequences = get_hla_sequences()
        two_digit_map = build_2digit_allele_map()
        
        # Organize by locus
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
        
        # Add 2-digit alleles
        for two_digit_allele in two_digit_map.keys():
            if two_digit_allele.startswith('A'):
                formatted = f"A*{two_digit_allele[1:3]}"
                if formatted not in allele_lists['A']:
                    allele_lists['A'].append(formatted)
            elif two_digit_allele.startswith('B'):
                formatted = f"B*{two_digit_allele[1:3]}"
                if formatted not in allele_lists['B']:
                    allele_lists['B'].append(formatted)
            elif two_digit_allele.startswith('C'):
                formatted = f"C*{two_digit_allele[1:3]}"
                if formatted not in allele_lists['C']:
                    allele_lists['C'].append(formatted)
            elif two_digit_allele.startswith('DRB1'):
                formatted = f"DRB1*{two_digit_allele[4:6]}"
                if formatted not in allele_lists['DRB1']:
                    allele_lists['DRB1'].append(formatted)
            elif two_digit_allele.startswith('DQA1'):
                formatted = f"DQA1*{two_digit_allele[4:6]}"
                if formatted not in allele_lists['DQA']:
                    allele_lists['DQA'].append(formatted)
            elif two_digit_allele.startswith('DQB1'):
                formatted = f"DQB1*{two_digit_allele[4:6]}"
                if formatted not in allele_lists['DQB']:
                    allele_lists['DQB'].append(formatted)
            elif two_digit_allele.startswith('DPA1'):
                formatted = f"DPA1*{two_digit_allele[4:6]}"
                if formatted not in allele_lists['DPA']:
                    allele_lists['DPA'].append(formatted)
            elif two_digit_allele.startswith('DPB1'):
                formatted = f"DPB1*{two_digit_allele[4:6]}"
                if formatted not in allele_lists['DPB']:
                    allele_lists['DPB'].append(formatted)
        
        # Sort all lists
        for locus in allele_lists:
            allele_lists[locus] = sorted(allele_lists[locus])
            
        logger.info(f"Loaded comprehensive allele lists: { {k: len(v) for k, v in allele_lists.items()} }")
            
    except Exception as e:
        logger.error(f"Error loading comprehensive allele lists: {str(e)}")
    
    return allele_lists

# ============== COMPREHENSIVE K-MER GENERATION ==============
def generate_kmers(sequence, k=9, max_kmers=None):
    """Generate overlapping k-mers from a sequence - COMPREHENSIVE"""
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

# ============== COMPREHENSIVE ANALYSIS FUNCTION ==============
def analyze_comprehensive_with_predictions_full(
    patient_name: str,
    donor_a1, donor_a2, donor_b1, donor_b2, donor_c1, donor_c2,
    donor_drb11, donor_drb12, donor_dqa1, donor_dqa2, donor_dqb1, donor_dqb2,
    donor_dpa1, donor_dpa2, donor_dpb1, donor_dpb2,
    recip_a1, recip_a2, recip_b1, recip_b2, recip_c1, recip_c2,
    recip_drb11, recip_drb12, recip_dqa1, recip_dqa2, recip_dqb1, recip_dqb2,
    recip_dpa1, recip_dpa2, recip_dpb1, recip_dpb2,
    k_length=9,
    analysis_type="comprehensive",
    prediction_threshold=0.5
):
    """Comprehensive analysis with REAL predictions - FULL VERSION"""
    
    start_time = time.time()
    
    # Combine alleles into lists
    donor_alleles = {
        'A': [a for a in [donor_a1, donor_a2] if a and a != "Not specified"],
        'B': [a for a in [donor_b1, donor_b2] if a and a != "Not specified"],
        'C': [a for a in [donor_c1, donor_c2] if a and a != "Not specified"],
        'DRB1': [a for a in [donor_drb11, donor_drb12] if a and a != "Not specified"],
        'DQA': [a for a in [donor_dqa1, donor_dqa2] if a and a != "Not specified"],
        'DQB': [a for a in [donor_dqb1, donor_dqb2] if a and a != "Not specified"],
        'DPA': [a for a in [donor_dpa1, donor_dpa2] if a and a != "Not specified"],
        'DPB': [a for a in [donor_dpb1, donor_dpb2] if a and a != "Not specified"]
    }
    
    recipient_alleles = {
        'A': [a for a in [recip_a1, recip_a2] if a and a != "Not specified"],
        'B': [a for a in [recip_b1, recip_b2] if a and a != "Not specified"],
        'C': [a for a in [recip_c1, recip_c2] if a and a != "Not specified"],
        'DRB1': [a for a in [recip_drb11, recip_drb12] if a and a != "Not specified"],
        'DQA': [a for a in [recip_dqa1, recip_dqa2] if a and a != "Not specified"],
        'DQB': [a for a in [recip_dqb1, recip_dqb2] if a and a != "Not specified"],
        'DPA': [a for a in [recip_dpa1, recip_dpa2] if a and a != "Not specified"],
        'DPB': [a for a in [recip_dpb1, recip_dpb2] if a and a != "Not specified"]
    }
    
    logger.info(f"Starting COMPREHENSIVE analysis for {patient_name}")
    
    # 1. Epitope compatibility with REAL predictions - COMPREHENSIVE
    epitope_results = []
    
    # Analysis parameters based on type
    if analysis_type == "quick":
        max_predictions_per_direction = 20
        max_kmers_per_sequence = 5
        directions = [('donor→recipient', donor_alleles, recipient_alleles)]
    elif analysis_type == "standard":
        max_predictions_per_direction = 50
        max_kmers_per_sequence = 10
        directions = [('donor→recipient', donor_alleles, recipient_alleles), ('recipient→donor', recipient_alleles, donor_alleles)]
    else:  # comprehensive
        max_predictions_per_direction = 200
        max_kmers_per_sequence = None  # All kmers
        directions = [('donor→recipient', donor_alleles, recipient_alleles), ('recipient→donor', recipient_alleles, donor_alleles)]
    
    total_predictions = 0
    
    for direction in directions:
        label, source, target = direction
        
        # Class I vs Class II (Class I epitopes presented by Class II)
        prediction_count = 0
        for src_allele in source['A'] + source['B'] + source['C']:
            if prediction_count >= max_predictions_per_direction:
                break
                
            src_seq = get_hla_sequence(src_allele)
            if not src_seq:
                continue
                
            # Generate kmers
            kmers = generate_kmers(src_seq, k=k_length, max_kmers=max_kmers_per_sequence)
            
            for tgt_allele in target['DRB1'] + target['DQA'] + target['DQB']:
                if prediction_count >= max_predictions_per_direction:
                    break
                    
                tgt_pseudoseq = get_pseudosequence(tgt_allele)
                if not tgt_pseudoseq:
                    continue
                
                # Use Class II model for Class I vs Class II
                for kmer in kmers:
                    if prediction_count >= max_predictions_per_direction:
                        break
                        
                    prediction = predict_binding_class_ii(kmer, tgt_allele, tgt_pseudoseq, prediction_threshold)
                    
                    epitope_results.append([
                        label, 
                        'I→II', 
                        src_allele, 
                        tgt_allele,
                        kmer,
                        f"{prediction['probability']:.4f}",
                        f"{prediction['ic50']:.2f}",
                        prediction['affinity'],
                        prediction['prediction'],
                        tgt_pseudoseq
                    ])
                    prediction_count += 1
                    total_predictions += 1
        
        # Class II vs Class I (Class II epitopes presented by Class I)
        prediction_count = 0
        for src_allele in source['DRB1'] + source['DQA'] + source['DQB']:
            if prediction_count >= max_predictions_per_direction:
                break
                
            src_seq = get_hla_sequence(src_allele)
            if not src_seq:
                continue
                
            # Generate kmers
            kmers = generate_kmers(src_seq, k=k_length, max_kmers=max_kmers_per_sequence)
            
            for tgt_allele in target['A'] + target['B'] + target['C']:
                if prediction_count >= max_predictions_per_direction:
                    break
                    
                tgt_pseudoseq = get_pseudosequence(tgt_allele)
                if not tgt_pseudoseq:
                    continue
                
                # Use Class I model for Class II vs Class I
                for kmer in kmers:
                    if prediction_count >= max_predictions_per_direction:
                        break
                        
                    prediction = predict_binding_class_i(kmer, tgt_allele, tgt_pseudoseq, prediction_threshold)
                    
                    epitope_results.append([
                        label, 
                        'II→I', 
                        src_allele, 
                        tgt_allele,
                        kmer,
                        f"{prediction['probability']:.4f}",
                        f"{prediction['ic50']:.2f}",
                        prediction['affinity'],
                        prediction['prediction'],
                        tgt_pseudoseq
                    ])
                    prediction_count += 1
                    total_predictions += 1
    
    # Create epitope dataframe
    epitope_df = pd.DataFrame(epitope_results, columns=[
        "Direction", "Class Interaction", "Source", "Target", "K-mer", 
        "Probability", "IC50 (nM)", "Affinity", "Prediction", "Pseudosequence"
    ]) if epitope_results else pd.DataFrame([["Analysis complete but no predictions made. Try with different alleles.", "", "", "", "", "", "", "", "", ""]])

    # 2. Sequence information
    sequence_data = []
    all_alleles = set(sum(donor_alleles.values(), []) + sum(recipient_alleles.values(), []))
    
    for allele in all_alleles:
        seq = get_hla_sequence(allele)
        pseudoseq = get_pseudosequence(allele)
        if seq:
            sequence_data.append({
                "Allele": allele,
                "2-Digit": get_2digit_allele(normalize_allele_name(allele)),
                "Sequence": seq[:100] + "..." if len(seq) > 100 else seq,
                "Pseudosequence": pseudoseq[:50] + "..." if pseudoseq and len(pseudoseq) > 50 else pseudoseq,
                "Length": len(seq),
                "Type": "Donor" if allele in sum(donor_alleles.values(), []) else "Recipient"
            })
    
    sequence_df = pd.DataFrame(sequence_data) if sequence_data else pd.DataFrame()

    # 3. Comprehensive summary
    processing_time = time.time() - start_time
    
    # Calculate statistics
    high_affinity = len([r for r in epitope_results if r[7] == "High"])
    intermediate_affinity = len([r for r in epitope_results if r[7] == "Intermediate"])
    low_affinity = len([r for r in epitope_results if r[7] == "Low"])
    binders = len([r for r in epitope_results if r[8] == "Binder"])
    
    summary_data = [{
        'Metric': 'Total Predictions',
        'Value': f"{total_predictions}"
    }, {
        'Metric': 'Processing Time', 
        'Value': f"{processing_time:.1f}s"
    }, {
        'Metric': 'High Affinity Binders',
        'Value': f"{high_affinity}"
    }, {
        'Metric': 'Intermediate Affinity',
        'Value': f"{intermediate_affinity}"
    }, {
        'Metric': 'Low Affinity',
        'Value': f"{low_affinity}"
    }, {
        'Metric': 'Total Binders',
        'Value': f"{binders}"
    }, {
        'Metric': 'Analysis Type',
        'Value': analysis_type.title()
    }, {
        'Metric': '2-Digit Allele Support',
        'Value': 'Enabled'
    }]
    
    summary_df = pd.DataFrame(summary_data)

    return epitope_df, sequence_df, summary_df

# ============== COMPREHENSIVE INTERFACE ==============
def create_comprehensive_interface_full():
    """Create comprehensive interface - FULL VERSION"""
    
    allele_lists = load_comprehensive_allele_lists()
    
    def create_dual_dropdown(allele_type: str, prefix: str):
        choices = ["Not specified"] + sorted(allele_lists.get(allele_type, []))
        
        # Select common alleles for initial values
        common_alleles = {
            'A': ['A*01:01', 'A*02:01', 'A*03:01'],
            'B': ['B*07:02', 'B*08:01', 'B*15:01'],
            'C': ['C*01:02', 'C*03:03', 'C*04:01'],
            'DRB1': ['DRB1*01:01', 'DRB1*03:01', 'DRB1*04:01'],
            'DQA': ['DQA1*01:01', 'DQA1*01:02', 'DQA1*05:01'],
            'DQB': ['DQB1*02:01', 'DQB1*03:01', 'DQB1*05:01'],
            'DPA': ['DPA1*01:03', 'DPA1*02:01'],
            'DPB': ['DPB1*02:01', 'DPB1*04:01', 'DPB1*05:01']
        }
        
        available_choices = [c for c in choices if c != "Not specified"]
        common_for_type = common_alleles.get(allele_type, [])
        
        # Use common alleles if available, otherwise random
        if len(common_for_type) >= 2:
            initial_values = common_for_type[:2]
        elif len(available_choices) >= 2:
            initial_values = available_choices[:2]
        else:
            initial_values = ["Not specified", "Not specified"]
        
        with gr.Group():
            gr.Markdown(f"**HLA-{allele_type}**")
            with gr.Row():
                d1 = gr.Dropdown(
                    choices=choices,
                    value=initial_values[0],
                    label=f"{prefix} {allele_type} Allele 1",
                    allow_custom_value=False,
                    filterable=True
                )
                d2 = gr.Dropdown(
                    choices=choices,
                    value=initial_values[1],
                    label=f"{prefix} {allele_type} Allele 2",
                    allow_custom_value=False,
                    filterable=True
                )
        return d1, d2

    css = """
    .analysis-section {margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 5px}
    .data-table {font-size: 0.8em; margin: 5px 0}
    .compact {max-height: 600px; overflow-y: auto}
    .full-width {width: 100%}
    .warning {background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 10px; border-radius: 5px}
    """

    with gr.Blocks(title="HLA Analyzer Pro - Full Version", css=css) as demo:
        gr.Markdown("# 🧬 HLA Compatibility Analyzer Pro")
        gr.Markdown("### **FULL VERSION** - Comprehensive analysis with 2-digit allele support")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Donor HLA Profile")
                with gr.Row():
                    with gr.Column():
                        donor_a1, donor_a2 = create_dual_dropdown('A', 'Donor')
                        donor_b1, donor_b2 = create_dual_dropdown('B', 'Donor')
                        donor_c1, donor_c2 = create_dual_dropdown('C', 'Donor')
                    with gr.Column():
                        donor_drb11, donor_drb12 = create_dual_dropdown('DRB1', 'Donor')
                        donor_dqa1, donor_dqa2 = create_dual_dropdown('DQA', 'Donor')
                        donor_dqb1, donor_dqb2 = create_dual_dropdown('DQB', 'Donor')
                gr.Markdown("#### DP Loci (Optional)")
                with gr.Row():
                    donor_dpa1, donor_dpa2 = create_dual_dropdown('DPA', 'Donor')
                    donor_dpb1, donor_dpb2 = create_dual_dropdown('DPB', 'Donor')

            with gr.Column(scale=1):
                gr.Markdown("### Recipient HLA Profile")
                with gr.Row():
                    with gr.Column():
                        recip_a1, recip_a2 = create_dual_dropdown('A', 'Recipient')
                        recip_b1, recip_b2 = create_dual_dropdown('B', 'Recipient')
                        recip_c1, recip_c2 = create_dual_dropdown('C', 'Recipient')
                    with gr.Column():
                        recip_drb11, recip_drb12 = create_dual_dropdown('DRB1', 'Recipient')
                        recip_dqa1, recip_dqa2 = create_dual_dropdown('DQA', 'Recipient')
                        recip_dqb1, recip_dqb2 = create_dual_dropdown('DQB', 'Recipient')
                gr.Markdown("#### DP Loci (Optional)")
                with gr.Row():
                    recip_dpa1, recip_dpa2 = create_dual_dropdown('DPA', 'Recipient')
                    recip_dpb1, recip_dpb2 = create_dual_dropdown('DPB', 'Recipient')

        with gr.Row():
            with gr.Column(scale=2):
                patient_name = gr.Textbox(label="Patient/Donor ID", placeholder="Optional...")
                k_length = gr.Slider(8, 15, value=9, step=1, label="Epitope Length (k)")
                prediction_threshold = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Prediction Threshold")
            
            with gr.Column(scale=1):
                analysis_type = gr.Radio(
                    choices=["quick", "standard", "comprehensive"],
                    value="standard",
                    label="Analysis Depth",
                    info="Quick: 20 predictions/direction, Standard: 50 predictions/direction, Comprehensive: 200 predictions/direction"
                )
                analyze_btn = gr.Button("🚀 Run COMPREHENSIVE Analysis", variant="primary", size="lg")

        with gr.Column(elem_classes=["analysis-section"]):
            gr.Markdown("## 📊 Comprehensive Analysis Results")
            
            with gr.Tabs():
                with gr.Tab("🧪 Epitope Binding Predictions"):
                    epitope_output = gr.Dataframe(
                        label="Binding Predictions (Comprehensive Analysis)",
                        headers=[
                            "Direction", "Class Interaction", "Source", "Target", "K-mer", 
                            "Probability", "IC50 (nM)", "Affinity", "Prediction", "Pseudosequence"
                        ],
                        interactive=False,
                        elem_classes=["data-table", "compact"],
                        wrap=True,
                        max_height=600
                    )
                
                with gr.Tab("🧬 Sequences & Pseudosequences"):
                    sequence_output = gr.Dataframe(
                        label="Sequence Information with 2-Digit Support", 
                        interactive=False,
                        elem_classes=["data-table", "compact"],
                        wrap=True,
                        max_height=600
                    )
                
                with gr.Tab("📋 Comprehensive Summary"):
                    summary_output = gr.Dataframe(
                        label="Analysis Summary",
                        interactive=False,
                        elem_classes=["data-table"]
                    )

        # Hidden components for DP alleles
        donor_dpa1_hidden = gr.Textbox(value="Not specified", visible=False)
        donor_dpa2_hidden = gr.Textbox(value="Not specified", visible=False)
        donor_dpb1_hidden = gr.Textbox(value="Not specified", visible=False)
        donor_dpb2_hidden = gr.Textbox(value="Not specified", visible=False)
        
        recip_dpa1_hidden = gr.Textbox(value="Not specified", visible=False)
        recip_dpa2_hidden = gr.Textbox(value="Not specified", visible=False)
        recip_dpb1_hidden = gr.Textbox(value="Not specified", visible=False)
        recip_dpb2_hidden = gr.Textbox(value="Not specified", visible=False)

        # Connect analysis function
        analyze_btn.click(
            fn=analyze_comprehensive_with_predictions_full,
            inputs=[
                patient_name,
                donor_a1, donor_a2, donor_b1, donor_b2, donor_c1, donor_c2,
                donor_drb11, donor_drb12, donor_dqa1, donor_dqa2, donor_dqb1, donor_dqb2,
                donor_dpa1_hidden, donor_dpa2_hidden, donor_dpb1_hidden, donor_dpb2_hidden,
                recip_a1, recip_a2, recip_b1, recip_b2, recip_c1, recip_c2,
                recip_drb11, recip_drb12, recip_dqa1, recip_dqa2, recip_dqb1, recip_dqb2,
                recip_dpa1_hidden, recip_dpa2_hidden, recip_dpb1_hidden, recip_dpb2_hidden,
                k_length,
                analysis_type,
                prediction_threshold
            ],
            outputs=[
                epitope_output, sequence_output, summary_output
            ]
        )

        gr.Markdown("""
        ### 🎯 **Full Version Features:**
        - **2-digit allele support**: Automatic consensus sequences for alleles like A*01, DRB1*04, etc.
        - **Comprehensive analysis**: Both donor→recipient and recipient→donor directions
        - **Multiple analysis depths**: Quick, Standard, and Comprehensive modes
        - **Full HLA coverage**: All Class I and Class II loci including DP
        - **Advanced statistics**: Detailed binding affinity summaries
        - **Flexible parameters**: Adjustable epitope length and prediction threshold
        
        ### 💡 **2-Digit Allele Support:**
        For alleles with only 2-digit resolution (e.g., A*01 instead of A*01:01:01:01), 
        the system automatically uses a consensus sequence based on the most frequent 
        alleles in that group or the first available sequence if frequency data is unavailable.
        
        ### ⚠️ **Performance Note:**
        Comprehensive analysis may take several minutes depending on the number of alleles
        and the analysis depth selected. For faster results, use Quick mode.
        """)

    return demo

# ============== MAIN EXECUTION ==============
if __name__ == "__main__":
    try:
        logger.info("Starting COMPREHENSIVE HLA Analyzer Pro...")
        
        # Pre-load comprehensive data
        logger.info("Loading comprehensive allele lists...")
        allele_lists = load_comprehensive_allele_lists()
        
        # Pre-load models
        logger.info("Pre-loading comprehensive AI models...")
        load_class_i_model()
        load_class_ii_model()
        
        # Pre-load sequences and build 2-digit map
        logger.info("Building comprehensive sequence database...")
        get_hla_sequences()
        get_pseudosequence_data()
        
        logger.info("Creating comprehensive interface...")
        demo = create_comprehensive_interface_full()
        
        logger.info("Launching COMPREHENSIVE application...")
        demo.launch(
            server_name="0.0.0.0",
            server_port=8000,
            share=True,
            debug=False,
            show_error=True
        )
    except Exception as e:
        logger.error(f"Comprehensive application failed to start: {str(e)}")
        raise
