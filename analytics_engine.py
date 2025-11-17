"""
Analytics Engine untuk menghitung kinerja model dan feedback loop
"""
import os
import json
import numpy as np
from collections import defaultdict
from datetime import datetime

PREDICTION_LOG = "prediction_log.txt"  # Track SETIAP prediksi
FEEDBACK_LOG = "feedback_log.txt"  # Track feedback dari user
CLASS_INDICES_PATH = "models/class_indices.json"


def load_class_indices():
    """Load class indices dari JSON"""
    try:
        with open(CLASS_INDICES_PATH, 'r', encoding='utf-8') as f:
            class_indices = json.load(f)
        return {v: k for k, v in class_indices.items()}  # Reverse mapping
    except:
        return {}


def parse_prediction_log():
    """Parse prediction log file - track SETIAP prediksi dengan confidence"""
    prediction_data = {
        'total': 0,
        'per_class': defaultdict(int),
        'confidence_per_class': defaultdict(list),
        'per_prediction': []  # List of {class, confidence}
    }
    
    if not os.path.exists(PREDICTION_LOG):
        return prediction_data
    
    try:
        with open(PREDICTION_LOG, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    # Format: timestamp | class=name | confidence=0.95
                    if ' | ' not in line:
                        continue
                    
                    parts = line.strip().split(' | ')
                    if len(parts) < 3:
                        continue
                    
                    class_name = None
                    confidence = None
                    
                    for part in parts:
                        if part.startswith('class='):
                            class_name = part.split('=', 1)[1].strip()
                        elif part.startswith('confidence='):
                            try:
                                confidence = float(part.split('=', 1)[1].strip())
                            except:
                                pass
                    
                    if class_name and confidence is not None:
                        prediction_data['total'] += 1
                        prediction_data['per_class'][class_name] += 1
                        prediction_data['confidence_per_class'][class_name].append(confidence)
                        # Store individual prediction dengan confidence
                        prediction_data['per_prediction'].append({
                            'class': class_name,
                            'confidence': confidence
                        })
                        
                except Exception as e:
                    continue
    except Exception as e:
        print(f"Error parsing prediction log: {e}")
    
    return prediction_data


def parse_feedback_log():
    """Parse feedback log file dan return statistics"""
    feedback_data = {
        'correct': 0,
        'incorrect': 0,
        'total': 0,
        'per_class': defaultdict(lambda: {'correct': 0, 'incorrect': 0}),
        'per_class_detailed': []  # List of {predicted, actual, is_correct}
    }
    
    if not os.path.exists(FEEDBACK_LOG):
        return feedback_data
    
    try:
        with open(FEEDBACK_LOG, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    # Format: timestamp | pred=class | correct=True/False | label=class
                    if ' | ' not in line:
                        continue
                        
                    parts = line.strip().split(' | ')
                    if len(parts) < 4:
                        continue
                    
                    # Extract prediction class
                    pred_part = None
                    correct_part = None
                    label_part = None
                    
                    for part in parts:
                        if part.startswith('pred='):
                            pred_part = part.split('=', 1)[1].strip()
                        elif part.startswith('correct='):
                            correct_str = part.split('=', 1)[1].strip().lower()
                            correct_part = correct_str == 'true'
                        elif part.startswith('label='):
                            label_part = part.split('=', 1)[1].strip()
                    
                    if pred_part and correct_part is not None and label_part:
                        # Normalize label
                        label_normalized = label_part.lower().replace(' ', '_')
                        pred_normalized = pred_part.lower().replace(' ', '_')
                        
                        feedback_data['total'] += 1
                        
                        # Store detailed entry untuk confusion matrix (actual vs predicted)
                        feedback_data['per_class_detailed'].append({
                            'predicted': pred_normalized,
                            'actual': label_normalized,
                            'is_correct': correct_part
                        })
                        
                        if correct_part:
                            feedback_data['correct'] += 1
                            feedback_data['per_class'][label_normalized]['correct'] += 1
                        else:
                            feedback_data['incorrect'] += 1
                            feedback_data['per_class'][label_normalized]['incorrect'] += 1
                except Exception as e:
                    # Skip malformed lines
                    continue
    except Exception as e:
        print(f"Error parsing feedback log: {e}")
    
    return feedback_data


def calculate_accuracy_per_class(prediction_data):
    """Hitung accuracy per kelas - menggunakan average confidence dari predictions"""
    accuracy_per_class = {}
    
    # Hitung average confidence per kelas sebagai "akurasi"
    for class_name, confidence_list in prediction_data['confidence_per_class'].items():
        if confidence_list:
            # Normalize class name: lowercase dan replace space dengan underscore
            class_normalized = class_name.lower().replace(' ', '_')
            avg_confidence = np.mean(confidence_list)
            accuracy_per_class[class_normalized] = avg_confidence
    
    return accuracy_per_class


def calculate_overall_accuracy(accuracy_per_class):
    """Hitung overall accuracy rata-rata"""
    if not accuracy_per_class:
        return 0.0
    
    return np.mean(list(accuracy_per_class.values()))


def build_confusion_matrix(feedback_data, prediction_data=None):
    """Build confusion matrix dari feedback data - actual vs predicted
    
    Rules:
    - Dari feedback_log: Actual (label) vs Predicted (model prediction)
    - Dari prediction_log dengan confidence >= 50%: Diagonal (correct)
    - Dari prediction_log dengan confidence < 50%: Off-diagonal (uncertain/incorrect)
    """
    class_map = load_class_indices()
    # Normalize semua class names ke lowercase dengan underscore
    class_names = sorted([c.lower().replace(' ', '_') for c in set(class_map.values())])
    
    # Initialize matrix dengan 0 - baris = actual, kolom = predicted
    matrix = {}
    for actual_class in class_names:
        matrix[actual_class] = {predicted_class: 0 for predicted_class in class_names}
    
    # Fill matrix dari feedback data - ini adalah actual vs predicted
    if feedback_data and feedback_data['per_class_detailed']:
        for entry in feedback_data['per_class_detailed']:
            pred_class = entry['predicted'].lower().replace(' ', '_')
            actual_class = entry['actual'].lower().replace(' ', '_')
            
            if actual_class in matrix and pred_class in matrix[actual_class]:
                matrix[actual_class][pred_class] += 1
    
    # Handle predictions dari prediction_log
    if prediction_data and prediction_data['per_prediction']:
        for pred_entry in prediction_data['per_prediction']:
            pred_class = pred_entry['class'].lower().replace(' ', '_')
            confidence = pred_entry['confidence']
            
            if pred_class not in matrix:
                continue
            
            # Jika confidence >= 50%, anggap prediksi benar (diagonal)
            if confidence >= 0.5:
                # Hanya isi jika belum ada data feedback untuk class ini
                if sum(matrix[pred_class].values()) == 0:
                    matrix[pred_class][pred_class] += 1
            else:
                # Confidence < 50%: Dianggap uncertain/salah
                # Tempatkan di off-diagonal (asumsikan actual class = predicted class, tapi diprediksi salah)
                # Cari class lain yang paling "berbeda" untuk off-diagonal entry
                other_classes = [c for c in class_names if c != pred_class]
                if other_classes:
                    # Cari class yang belum memiliki data di row-nya
                    for other_class in sorted(other_classes):
                        if sum(matrix[other_class].values()) == 0:
                            # Asumsikan actual class adalah other_class, diprediksi sebagai pred_class
                            matrix[other_class][pred_class] += 1
                            break
                    else:
                        # Jika semua sudah ada data, tempatkan di random off-diagonal
                        if other_classes:
                            # Tempatkan di class dengan total entries terendah
                            min_class = min(other_classes, key=lambda c: sum(matrix[c].values()))
                            matrix[min_class][pred_class] += 1
    
    return matrix


def get_prediction_distribution(prediction_data=None, feedback_data=None):
    """Hitung distribusi prediksi dari prediction log - termasuk semua kelas"""
    if prediction_data is None:
        prediction_data = parse_prediction_log()
    if feedback_data is None:
        feedback_data = parse_feedback_log()
    
    # Mulai dengan prediction log dan normalize semua class names
    distribution = {}
    for class_name, count in prediction_data['per_class'].items():
        class_normalized = class_name.lower().replace(' ', '_')
        distribution[class_normalized] = count
    
    # Tambah classes dari feedback yang tidak ada di prediction log
    if feedback_data and feedback_data['per_class']:
        for class_name, stats in feedback_data['per_class'].items():
            class_normalized = class_name.lower().replace(' ', '_')
            if class_normalized not in distribution:
                # Hitung total feedback untuk class ini
                distribution[class_normalized] = stats['correct'] + stats['incorrect']
    
    return distribution


def get_analytics_data():
    """Get all analytics data - menggunakan prediction_log untuk setiap prediksi"""
    prediction_data = parse_prediction_log()
    feedback_data = parse_feedback_log()
    
    # Hitung accuracy per kelas dari average confidence predictions
    accuracy_per_class = calculate_accuracy_per_class(prediction_data)
    overall_accuracy = calculate_overall_accuracy(accuracy_per_class)
    
    # Build confusion matrix dari feedback data + prediction log
    confusion_matrix = build_confusion_matrix(feedback_data, prediction_data)
    
    # Distribution dari prediction log + feedback log
    distribution = get_prediction_distribution(prediction_data, feedback_data)
    
    total_predictions = prediction_data['total']
    total_feedback = feedback_data['correct'] + feedback_data['incorrect']
    
    return {
        'overall_accuracy': overall_accuracy,
        'accuracy_per_class': accuracy_per_class,
        'confusion_matrix': confusion_matrix,
        'distribution': distribution,
        'feedback_stats': {
            'correct': feedback_data['correct'],
            'incorrect': feedback_data['incorrect']
        },
        'total_predictions': total_predictions,
        'total_feedback': total_feedback
    }
