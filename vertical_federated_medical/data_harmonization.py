"""
Automated Data Harmonization for Vertical Federated Learning
Automatically detects and aligns schema, semantics, and modalities
across different healthcare organizations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import torch
from dataclasses import dataclass
from enum import Enum
import re
from difflib import SequenceMatcher
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataType(Enum):
    """Different types of data formats"""
    TABULAR = "tabular"
    IMAGES = "images"
    TEXT = "text"
    TIME_SERIES = "time_series"
    GRAPH = "graph"

class MedicalCodingSystem(Enum):
    """Different medical coding systems"""
    ICD10 = "icd10"
    ICD9 = "icd9"
    SNOMED = "snomed"
    CPT = "cpt"
    LOINC = "loinc"
    RXNORM = "rxnorm"

@dataclass
class SchemaMapping:
    """Schema mapping between different organizations"""
    source_column: str
    target_column: str
    confidence: float
    transformation: str
    data_type: str

@dataclass
class DataQualityMetrics:
    """Data quality assessment metrics"""
    completeness: float
    consistency: float
    accuracy: float
    timeliness: float
    validity: float
    overall_score: float

class MedicalDataHarmonizer:
    """Automated harmonization of medical data across organizations"""
    
    def __init__(self):
        self.schema_mappings = {}
        self.quality_metrics = {}
        self.coding_system_mappings = self._initialize_coding_mappings()
        self.medical_terminology = self._load_medical_terminology()
        
    def _initialize_coding_mappings(self) -> Dict[MedicalCodingSystem, Dict[str, str]]:
        """Initialize mappings between different medical coding systems"""
        return {
            MedicalCodingSystem.ICD10: {
                "diabetes": "E11",
                "hypertension": "I10", 
                "pneumonia": "J18",
                "myocardial_infarction": "I21"
            },
            MedicalCodingSystem.SNOMED: {
                "diabetes": "44054006",
                "hypertension": "38341003",
                "pneumonia": "233604007",
                "myocardial_infarction": "22298006"
            },
            MedicalCodingSystem.ICD9: {
                "diabetes": "250",
                "hypertension": "401",
                "pneumonia": "486",
                "myocardial_infarction": "410"
            }
        }
    
    def _load_medical_terminology(self) -> Dict[str, List[str]]:
        """Load medical terminology for semantic matching"""
        return {
            "diabetes": ["diabetes", "diabetic", "dm", "diabetes_mellitus", "t2dm", "type_2_diabetes"],
            "hypertension": ["hypertension", "htn", "high_blood_pressure", "elevated_bp", "bp_elevated"],
            "pneumonia": ["pneumonia", "pneumonitis", "lung_infection", "respiratory_infection"],
            "myocardial_infarction": ["mi", "heart_attack", "myocardial_infarction", "acute_mi", "ami"]
        }
    
    def detect_data_modalities(self, datasets: Dict[str, Any]) -> Dict[str, DataType]:
        """Detect data modalities across organizations"""
        modality_detection = {}
        
        for org_id, dataset in datasets.items():
            if isinstance(dataset, pd.DataFrame):
                # Analyze column types and content
                if self._has_image_columns(dataset):
                    modality_detection[org_id] = DataType.IMAGES
                elif self._has_text_columns(dataset):
                    modality_detection[org_id] = DataType.TEXT
                elif self._has_time_series_columns(dataset):
                    modality_detection[org_id] = DataType.TIME_SERIES
                else:
                    modality_detection[org_id] = DataType.TABULAR
            else:
                modality_detection[org_id] = DataType.TABULAR
        
        return modality_detection
    
    def _has_image_columns(self, df: pd.DataFrame) -> bool:
        """Check if dataset has image-related columns"""
        image_keywords = ['image', 'img', 'scan', 'xray', 'mri', 'ct', 'pathology', 'slide']
        return any(keyword in col.lower() for col in df.columns for keyword in image_keywords)
    
    def _has_text_columns(self, df: pd.DataFrame) -> bool:
        """Check if dataset has text columns"""
        text_columns = df.select_dtypes(include=['object']).columns
        return len(text_columns) > 0 and any(df[col].str.len().mean() > 10 for col in text_columns)
    
    def _has_time_series_columns(self, df: pd.DataFrame) -> bool:
        """Check if dataset has time series columns"""
        time_keywords = ['time', 'date', 'timestamp', 'hour', 'day', 'week', 'month']
        return any(keyword in col.lower() for col in df.columns for keyword in time_keywords)
    
    def analyze_schema_differences(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """Analyze schema differences across organizations"""
        schema_analysis = {}
        
        for org_id, dataset in datasets.items():
            analysis = {
                'columns': list(dataset.columns),
                'dtypes': dataset.dtypes.to_dict(),
                'shape': dataset.shape,
                'missing_values': dataset.isnull().sum().to_dict(),
                'unique_values': {col: dataset[col].nunique() for col in dataset.columns},
                'value_ranges': self._get_value_ranges(dataset),
                'categorical_values': self._get_categorical_values(dataset)
            }
            schema_analysis[org_id] = analysis
        
        return schema_analysis
    
    def _get_value_ranges(self, df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
        """Get value ranges for numerical columns"""
        ranges = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            ranges[col] = (df[col].min(), df[col].max())
        return ranges
    
    def _get_categorical_values(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Get unique values for categorical columns"""
        categorical = {}
        for col in df.select_dtypes(include=['object']).columns:
            categorical[col] = df[col].unique().tolist()[:10]  # Limit to first 10 values
        return categorical
    
    def create_schema_mappings(self, source_org: str, target_org: str, 
                              schema_analysis: Dict[str, Dict[str, Any]]) -> List[SchemaMapping]:
        """Create schema mappings between two organizations"""
        mappings = []
        
        source_schema = schema_analysis[source_org]
        target_schema = schema_analysis[target_org]
        
        source_columns = source_schema['columns']
        target_columns = target_schema['columns']
        
        for source_col in source_columns:
            best_match = None
            best_confidence = 0.0
            best_transformation = "direct"
            
            for target_col in target_columns:
                # Calculate similarity
                similarity = self._calculate_column_similarity(
                    source_col, target_col, 
                    source_schema, target_schema
                )
                
                if similarity > best_confidence and similarity > 0.6:  # Threshold
                    best_confidence = similarity
                    best_match = target_col
                    best_transformation = self._determine_transformation(
                        source_col, target_col, source_schema, target_schema
                    )
            
            if best_match:
                mapping = SchemaMapping(
                    source_column=source_col,
                    target_column=best_match,
                    confidence=best_confidence,
                    transformation=best_transformation,
                    data_type=source_schema['dtypes'][source_col]
                )
                mappings.append(mapping)
        
        return mappings
    
    def _calculate_column_similarity(self, col1: str, col2: str, 
                                   schema1: Dict, schema2: Dict) -> float:
        """Calculate similarity between two columns"""
        # String similarity
        string_sim = SequenceMatcher(None, col1.lower(), col2.lower()).ratio()
        
        # Semantic similarity (using medical terminology)
        semantic_sim = self._calculate_semantic_similarity(col1, col2)
        
        # Data type similarity
        dtype_sim = 1.0 if schema1['dtypes'][col1] == schema2['dtypes'][col2] else 0.5
        
        # Value range similarity (for numerical columns)
        range_sim = self._calculate_range_similarity(col1, col2, schema1, schema2)
        
        # Weighted combination
        total_similarity = (
            string_sim * 0.3 +
            semantic_sim * 0.4 +
            dtype_sim * 0.2 +
            range_sim * 0.1
        )
        
        return total_similarity
    
    def _calculate_semantic_similarity(self, col1: str, col2: str) -> float:
        """Calculate semantic similarity using medical terminology"""
        col1_lower = col1.lower()
        col2_lower = col2.lower()
        
        # Check if columns match any medical terminology
        for condition, terms in self.medical_terminology.items():
            col1_match = any(term in col1_lower for term in terms)
            col2_match = any(term in col2_lower for term in terms)
            
            if col1_match and col2_match:
                return 0.9  # High similarity for medical terms
        
        # Check for common medical prefixes/suffixes
        medical_prefixes = ['diag', 'dx', 'icd', 'snomed', 'cpt', 'loinc']
        medical_suffixes = ['_code', '_id', '_type', '_category']
        
        col1_medical = any(prefix in col1_lower for prefix in medical_prefixes) or \
                      any(suffix in col1_lower for suffix in medical_suffixes)
        col2_medical = any(prefix in col2_lower for prefix in medical_prefixes) or \
                      any(suffix in col2_lower for suffix in medical_suffixes)
        
        if col1_medical and col2_medical:
            return 0.7  # Medium similarity for medical codes
        
        return 0.3  # Low similarity for non-medical terms
    
    def _calculate_range_similarity(self, col1: str, col2: str, 
                                  schema1: Dict, schema2: Dict) -> float:
        """Calculate similarity based on value ranges"""
        if col1 not in schema1['value_ranges'] or col2 not in schema2['value_ranges']:
            return 0.5  # Neutral for non-numerical columns
        
        range1 = schema1['value_ranges'][col1]
        range2 = schema2['value_ranges'][col2]
        
        # Calculate overlap
        min_overlap = max(range1[0], range2[0])
        max_overlap = min(range1[1], range2[1])
        
        if max_overlap <= min_overlap:
            return 0.0  # No overlap
        
        overlap_size = max_overlap - min_overlap
        range1_size = range1[1] - range1[0]
        range2_size = range2[1] - range2[0]
        
        # Jaccard similarity
        union_size = range1_size + range2_size - overlap_size
        similarity = overlap_size / union_size if union_size > 0 else 0.0
        
        return similarity
    
    def _determine_transformation(self, source_col: str, target_col: str,
                                source_schema: Dict, target_schema: Dict) -> str:
        """Determine the transformation needed between columns"""
        source_dtype = source_schema['dtypes'][source_col]
        target_dtype = target_schema['dtypes'][target_col]
        
        if source_dtype == target_dtype:
            return "direct"
        elif source_dtype == 'object' and target_dtype in ['int64', 'float64']:
            return "encode_categorical"
        elif source_dtype in ['int64', 'float64'] and target_dtype == 'object':
            return "decode_numerical"
        elif source_dtype == 'int64' and target_dtype == 'float64':
            return "int_to_float"
        elif source_dtype == 'float64' and target_dtype == 'int64':
            return "float_to_int"
        else:
            return "custom_transformation"
    
    def harmonize_datasets(self, datasets: Dict[str, pd.DataFrame], 
                          mappings: Dict[str, List[SchemaMapping]]) -> Dict[str, pd.DataFrame]:
        """Apply schema mappings to harmonize datasets"""
        harmonized_datasets = {}
        
        for org_id, dataset in datasets.items():
            if org_id in mappings:
                org_mappings = mappings[org_id]
                harmonized_df = dataset.copy()
                
                # Apply column renames
                rename_dict = {mapping.source_column: mapping.target_column 
                             for mapping in org_mappings}
                harmonized_df = harmonized_df.rename(columns=rename_dict)
                
                # Apply transformations
                for mapping in org_mappings:
                    if mapping.transformation == "encode_categorical":
                        le = LabelEncoder()
                        harmonized_df[mapping.target_column] = le.fit_transform(
                            harmonized_df[mapping.target_column].astype(str)
                        )
                    elif mapping.transformation == "int_to_float":
                        harmonized_df[mapping.target_column] = harmonized_df[mapping.target_column].astype(float)
                    elif mapping.transformation == "float_to_int":
                        harmonized_df[mapping.target_column] = harmonized_df[mapping.target_column].astype(int)
                
                harmonized_datasets[org_id] = harmonized_df
            else:
                harmonized_datasets[org_id] = dataset
        
        return harmonized_datasets
    
    def assess_data_quality(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, DataQualityMetrics]:
        """Assess data quality across organizations"""
        quality_metrics = {}
        
        for org_id, dataset in datasets.items():
            metrics = DataQualityMetrics(
                completeness=self._calculate_completeness(dataset),
                consistency=self._calculate_consistency(dataset),
                accuracy=self._calculate_accuracy(dataset),
                timeliness=self._calculate_timeliness(dataset),
                validity=self._calculate_validity(dataset),
                overall_score=0.0
            )
            
            # Calculate overall score
            metrics.overall_score = (
                metrics.completeness * 0.25 +
                metrics.consistency * 0.25 +
                metrics.accuracy * 0.25 +
                metrics.timeliness * 0.15 +
                metrics.validity * 0.10
            )
            
            quality_metrics[org_id] = metrics
        
        return quality_metrics
    
    def _calculate_completeness(self, df: pd.DataFrame) -> float:
        """Calculate data completeness score"""
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        completeness = 1.0 - (missing_cells / total_cells)
        return completeness
    
    def _calculate_consistency(self, df: pd.DataFrame) -> float:
        """Calculate data consistency score"""
        # Check for duplicate rows
        duplicate_ratio = df.duplicated().sum() / len(df)
        
        # Check for inconsistent data types
        inconsistent_dtypes = 0
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if all values can be converted to numeric
                try:
                    pd.to_numeric(df[col], errors='raise')
                except:
                    inconsistent_dtypes += 1
        
        dtype_consistency = 1.0 - (inconsistent_dtypes / len(df.columns))
        consistency = (1.0 - duplicate_ratio) * 0.7 + dtype_consistency * 0.3
        
        return consistency
    
    def _calculate_accuracy(self, df: pd.DataFrame) -> float:
        """Calculate data accuracy score (simplified)"""
        # Check for outliers in numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        outlier_ratio = 0.0
        
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
            outlier_ratio += outliers / len(df)
        
        outlier_ratio /= len(numerical_cols) if len(numerical_cols) > 0 else 1
        accuracy = 1.0 - min(outlier_ratio, 1.0)
        
        return accuracy
    
    def _calculate_timeliness(self, df: pd.DataFrame) -> float:
        """Calculate data timeliness score"""
        # Check for date columns and their recency
        date_cols = df.select_dtypes(include=['datetime64']).columns
        if len(date_cols) == 0:
            return 0.5  # Neutral if no date columns
        
        timeliness_scores = []
        for col in date_cols:
            if df[col].notna().any():
                max_date = df[col].max()
                if pd.notna(max_date):
                    days_old = (pd.Timestamp.now() - max_date).days
                    # Score decreases with age (max 365 days)
                    score = max(0, 1.0 - (days_old / 365))
                    timeliness_scores.append(score)
        
        return np.mean(timeliness_scores) if timeliness_scores else 0.5
    
    def _calculate_validity(self, df: pd.DataFrame) -> float:
        """Calculate data validity score"""
        # Check for valid values in categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        validity_scores = []
        
        for col in categorical_cols:
            # Check for empty strings or null values
            valid_values = df[col].notna() & (df[col] != '') & (df[col] != ' ')
            validity_ratio = valid_values.sum() / len(df)
            validity_scores.append(validity_ratio)
        
        return np.mean(validity_scores) if validity_scores else 1.0
    
    def generate_harmonization_report(self, 
                                    schema_analysis: Dict[str, Dict[str, Any]],
                                    mappings: Dict[str, List[SchemaMapping]],
                                    quality_metrics: Dict[str, DataQualityMetrics]) -> Dict[str, Any]:
        """Generate comprehensive harmonization report"""
        report = {
            "summary": {
                "total_organizations": len(schema_analysis),
                "total_mappings": sum(len(m) for m in mappings.values()),
                "average_quality_score": np.mean([m.overall_score for m in quality_metrics.values()])
            },
            "schema_analysis": schema_analysis,
            "mappings": {
                org_id: [
                    {
                        "source": m.source_column,
                        "target": m.target_column,
                        "confidence": m.confidence,
                        "transformation": m.transformation
                    } for m in org_mappings
                ] for org_id, org_mappings in mappings.items()
            },
            "quality_metrics": {
                org_id: {
                    "completeness": m.completeness,
                    "consistency": m.consistency,
                    "accuracy": m.accuracy,
                    "timeliness": m.timeliness,
                    "validity": m.validity,
                    "overall_score": m.overall_score
                } for org_id, m in quality_metrics.items()
            }
        }
        
        return report

# Example usage
if __name__ == "__main__":
    # Create sample datasets
    np.random.seed(42)
    
    # Hospital A data
    hospital_a = pd.DataFrame({
        'patient_id': range(100),
        'age': np.random.randint(18, 80, 100),
        'blood_pressure': np.random.normal(120, 20, 100),
        'diabetes_diagnosis': np.random.choice(['yes', 'no'], 100),
        'xray_image_path': [f'/images/xray_{i}.jpg' for i in range(100)]
    })
    
    # Hospital B data (different schema)
    hospital_b = pd.DataFrame({
        'id': range(100),
        'age_years': np.random.randint(18, 80, 100),
        'bp_systolic': np.random.normal(120, 20, 100),
        'diabetes_status': np.random.choice([1, 0], 100),
        'chest_xray': [f'/scans/chest_{i}.dcm' for i in range(100)]
    })
    
    # Clinic data
    clinic = pd.DataFrame({
        'patient_id': range(100),
        'age': np.random.randint(18, 80, 100),
        'medication_list': [f'med_{i}' for i in range(100)],
        'lab_results': np.random.normal(7.0, 1.0, 100)
    })
    
    datasets = {
        'hospital_a': hospital_a,
        'hospital_b': hospital_b,
        'clinic': clinic
    }
    
    # Initialize harmonizer
    harmonizer = MedicalDataHarmonizer()
    
    # Detect modalities
    modalities = harmonizer.detect_data_modalities(datasets)
    print(f"Detected modalities: {modalities}")
    
    # Analyze schemas
    schema_analysis = harmonizer.analyze_schema_differences(datasets)
    print(f"Schema analysis completed for {len(schema_analysis)} organizations")
    
    # Create mappings
    mappings = {}
    for org_id in ['hospital_a', 'hospital_b']:
        if org_id != 'hospital_a':  # Use hospital_a as reference
            mappings[org_id] = harmonizer.create_schema_mappings(
                org_id, 'hospital_a', schema_analysis
            )
    
    print(f"Created {sum(len(m) for m in mappings.values())} schema mappings")
    
    # Assess quality
    quality_metrics = harmonizer.assess_data_quality(datasets)
    print(f"Quality assessment completed")
    
    # Generate report
    report = harmonizer.generate_harmonization_report(
        schema_analysis, mappings, quality_metrics
    )
    
    print("Data harmonization completed successfully!")
    print(f"Average quality score: {report['summary']['average_quality_score']:.3f}")

