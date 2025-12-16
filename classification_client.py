"""
Classification Client Module
Interface to communicate with the MAZIK Classification API
"""

import requests
import base64
import io
from PIL import Image
import numpy as np
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)

class ClassificationClient:
    """Client for MAZIK Medical Classification API"""
    
    def __init__(self, api_url: str = "http://localhost:5002"):
        """
        Initialize classification client
        
        Args:
            api_url: Base URL of the classification API
        """
        self.api_url = api_url.rstrip('/')
        self.brain_endpoint = f"{self.api_url}/predict/brain_tumor"
        self.mammography_endpoint = f"{self.api_url}/predict/mammography"
        
    def check_health(self) -> tuple[bool, Optional[Dict]]:
        """
        Check if classification API is available
        
        Returns:
            Tuple of (is_healthy, health_data)
        """
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code == 200:
                return True, response.json()
            return False, None
        except Exception as e:
            logger.error(f"Classification API health check failed: {str(e)}")
            return False, None
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """
        Convert PIL Image to base64 string
        
        Args:
            image: PIL Image object
            
        Returns:
            Base64 encoded string
        """
        buffered = io.BytesIO()
        # Convert to RGB if necessary
        if image.mode in ('RGBA', 'LA', 'P'):
            image = image.convert('RGB')
        elif image.mode == 'L':
            image = image.convert('RGB')
        
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str
    
    def classify_brain_tumor(self, image: Image.Image, timeout: int = 30) -> Dict:
        """
        Classify brain tumor image
        
        Args:
            image: PIL Image object
            timeout: Request timeout in seconds
            
        Returns:
            Classification result dictionary
        """
        try:
            # Convert image to base64
            img_base64 = self._image_to_base64(image)
            
            # Prepare request
            payload = {"image": img_base64}
            
            # Make request
            response = requests.post(
                self.brain_endpoint,
                json=payload,
                timeout=timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('status') == 'success':
                    return {
                        'success': True,
                        'prediction': result['prediction'],
                        'model_type': 'brain_tumor'
                    }
                else:
                    return {
                        'success': False,
                        'error': result.get('error', 'Unknown error'),
                        'model_type': 'brain_tumor'
                    }
            else:
                return {
                    'success': False,
                    'error': f'API returned status code {response.status_code}',
                    'model_type': 'brain_tumor'
                }
                
        except Exception as e:
            logger.error(f"Brain tumor classification failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'model_type': 'brain_tumor'
            }
    
    def classify_mammography(self, image: Image.Image, timeout: int = 30) -> Dict:
        """
        Classify mammography image
        
        Args:
            image: PIL Image object
            timeout: Request timeout in seconds
            
        Returns:
            Classification result dictionary
        """
        try:
            # Convert image to base64
            img_base64 = self._image_to_base64(image)
            
            # Prepare request
            payload = {"image": img_base64}
            
            # Make request
            response = requests.post(
                self.mammography_endpoint,
                json=payload,
                timeout=timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('status') == 'success':
                    return {
                        'success': True,
                        'prediction': result['prediction'],
                        'model_type': 'mammography'
                    }
                else:
                    return {
                        'success': False,
                        'error': result.get('error', 'Unknown error'),
                        'model_type': 'mammography'
                    }
            else:
                return {
                    'success': False,
                    'error': f'API returned status code {response.status_code}',
                    'model_type': 'mammography'
                }
                
        except Exception as e:
            logger.error(f"Mammography classification failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'model_type': 'mammography'
            }
    
    def auto_classify(self, image: Image.Image, image_type: str = 'auto', timeout: int = 30) -> Dict:
        """
        Automatically classify image based on type or try both models
        
        Args:
            image: PIL Image object
            image_type: 'brain', 'mammography', or 'auto' to try both
            timeout: Request timeout in seconds
            
        Returns:
            Classification result dictionary
        """
        if image_type.lower() == 'brain':
            return self.classify_brain_tumor(image, timeout)
        elif image_type.lower() in ['mammography', 'breast']:
            return self.classify_mammography(image, timeout)
        else:
            # Auto mode: Try brain first, then mammography
            brain_result = self.classify_brain_tumor(image, timeout)
            
            if brain_result['success']:
                pred = brain_result['prediction']
                # Check if confident prediction
                if pred.get('confidence', 0) >= 0.7:
                    return brain_result
            
            # Try mammography
            mammo_result = self.classify_mammography(image, timeout)
            
            # Return the more confident result
            if brain_result['success'] and mammo_result['success']:
                brain_conf = brain_result['prediction'].get('confidence', 0)
                mammo_conf = mammo_result['prediction'].get('confidence', 0)
                
                if mammo_conf > brain_conf:
                    return mammo_result
                return brain_result
            elif brain_result['success']:
                return brain_result
            elif mammo_result['success']:
                return mammo_result
            else:
                return {
                    'success': False,
                    'error': 'Both classification models failed',
                    'model_type': 'auto'
                }
    
    def classify_batch(self, images: List[Image.Image], image_type: str = 'auto', timeout: int = 30) -> List[Dict]:
        """
        Classify multiple images
        
        Args:
            images: List of PIL Image objects
            image_type: 'brain', 'mammography', or 'auto'
            timeout: Request timeout in seconds per image
            
        Returns:
            List of classification result dictionaries
        """
        results = []
        for idx, image in enumerate(images):
            try:
                result = self.auto_classify(image, image_type, timeout)
                result['image_index'] = idx
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to classify image {idx}: {str(e)}")
                results.append({
                    'success': False,
                    'error': str(e),
                    'image_index': idx,
                    'model_type': image_type
                })
        
        return results
    
    def intelligent_classify_from_metadata(self, image: Image.Image, metadata: Dict, timeout: int = 30) -> Dict:
        """
        Intelligently classify image based on DICOM metadata
        
        Args:
            image: PIL Image object
            metadata: DICOM metadata dictionary
            timeout: Request timeout in seconds
            
        Returns:
            Classification result dictionary
        """
        # Extract relevant metadata
        modality = metadata.get('Modality', '').upper()
        body_part = metadata.get('BodyPartExamined', '').upper()
        study_desc = metadata.get('StudyDescription', '').upper()
        series_desc = metadata.get('SeriesDescription', '').upper()
        
        # Combine all text for analysis
        metadata_text = f"{modality} {body_part} {study_desc} {series_desc}"
        
        # Brain-related keywords
        brain_keywords = [
            'BRAIN', 'HEAD', 'CRANIAL', 'CEREBRAL', 'NEURO',
            'MR', 'MRI', 'CT HEAD', 'SKULL', 'INTRACRANIAL',
            'GLIOMA', 'MENINGIOMA', 'PITUITARY', 'TUMOR'
        ]
        
        # Breast/Mammography-related keywords
        breast_keywords = [
            'BREAST', 'MAMMO', 'MAMMOGRAPHY', 'MG',
            'TOMOSYNTHESIS', 'CHEST WALL', 'PECTORAL'
        ]
        
        # Count keyword matches
        brain_score = sum(1 for keyword in brain_keywords if keyword in metadata_text)
        breast_score = sum(1 for keyword in breast_keywords if keyword in metadata_text)
        
        # Decide which model to use
        if brain_score > breast_score:
            logger.info(f"Metadata suggests brain imaging (score: {brain_score}). Using brain tumor model.")
            return self.classify_brain_tumor(image, timeout)
        elif breast_score > brain_score:
            logger.info(f"Metadata suggests breast imaging (score: {breast_score}). Using mammography model.")
            return self.classify_mammography(image, timeout)
        else:
            # If unclear, use auto mode
            logger.info("Metadata unclear. Using auto-detection mode.")
            return self.auto_classify(image, 'auto', timeout)
    
    def format_classification_for_prompt(self, classification_result: Dict) -> str:
        """
        Format classification result for inclusion in reporting prompt
        
        Args:
            classification_result: Classification result dictionary
            
        Returns:
            Formatted string for prompt
        """
        if not classification_result.get('success'):
            return ""
        
        pred = classification_result.get('prediction', {})
        model_type = classification_result.get('model_type', 'unknown')
        
        predicted_class = pred.get('predicted_class', 'Unknown')
        confidence = pred.get('confidence', 0) * 100
        class_probs = pred.get('class_probabilities', {})
        
        # Format output
        output = f"\n**Pre-Analysis Classification Results:**\n"
        output += f"- Model Type: {model_type.replace('_', ' ').title()}\n"
        output += f"- Predicted Class: **{predicted_class.replace('_', ' ')}**\n"
        output += f"- Confidence: **{confidence:.1f}%**\n"
        
        if class_probs:
            output += f"\n**Class Probabilities:**\n"
            for class_name, prob in sorted(class_probs.items(), key=lambda x: x[1], reverse=True):
                output += f"- {class_name.replace('_', ' ')}: {prob*100:.1f}%\n"
        
        output += "\n*Please consider this classification in your detailed radiological analysis.*\n\n"
        
        return output

