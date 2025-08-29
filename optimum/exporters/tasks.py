# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tasks manager for ONNX exports."""

from typing import Dict, Any, Optional
from functools import partial


class TasksManager:
    """Minimal TasksManager implementation for optimum-onnx package."""
    
    _SUPPORTED_MODEL_TYPE = {}
    _TIMM_SUPPORTED_MODEL_TYPE = {}
    _SENTENCE_TRANSFORMERS_SUPPORTED_MODEL_TYPE = {}
    _DIFFUSERS_SUPPORTED_MODEL_TYPE = {}
    
    @staticmethod
    def get_all_tasks():
        """Return list of all supported tasks."""
        return [
            "feature-extraction",
            "text-classification", 
            "token-classification",
            "text-generation",
            "text-generation-with-past",
            "text2text-generation",
            "text2text-generation-with-past",
            "fill-mask",
            "question-answering",
            "summarization",
            "translation",
            "automatic-speech-recognition",
            "image-classification",
            "object-detection",
            "image-segmentation",
            "image-to-text",
            "text-to-image",
            "inpainting",
        ]
    
    @staticmethod
    def map_from_synonym(task: str) -> str:
        """Map task synonyms to canonical task names."""
        # Common task synonyms
        synonyms = {
            "sentence-similarity": "feature-extraction",
            "text-similarity": "feature-extraction",
            "zero-shot-classification": "text-classification",
            "causal-lm": "text-generation",
            "causal-lm-with-past": "text-generation-with-past",
            "seq2seq-lm": "text2text-generation",
            "seq2seq-lm-with-past": "text2text-generation-with-past",
            "masked-lm": "fill-mask",
            "asr": "automatic-speech-recognition",
            "speech-to-text": "automatic-speech-recognition",
        }
        return synonyms.get(task, task)
    
    @staticmethod
    def determine_framework(model_name_or_path: str, **kwargs) -> str:
        """Determine the framework from model artifacts."""
        # Simple heuristic - assume PyTorch for now
        return "pt"
    
    @staticmethod
    def infer_library_from_model(model_name_or_path: str, **kwargs) -> str:
        """Infer the library name from model."""
        # Default to transformers for now
        return "transformers"
    
    @staticmethod
    def infer_task_from_model(model_name_or_path: str, **kwargs) -> str:
        """Infer the task from model configuration."""
        # Simple fallback - would need actual model inspection in production
        return "feature-extraction"
    
    @staticmethod
    def get_supported_tasks_for_model_type(model_type: str, exporter: str = "onnx", **kwargs) -> Dict[str, Any]:
        """Get supported tasks for a given model type."""
        # Minimal implementation - return common tasks
        common_tasks = {
            "feature-extraction": None,
            "text-classification": None,
            "token-classification": None,
            "text-generation": None,
            "text-generation-with-past": None,
        }
        return common_tasks
    
    @staticmethod
    def get_model_from_task(task: str, model_name_or_path: str, **kwargs):
        """Load model for the specified task."""
        from transformers import AutoModel, AutoTokenizer
        
        # Determine device
        device = kwargs.get("device", "cpu")
        torch_dtype = kwargs.get("torch_dtype", None)
        
        # Load model
        model = AutoModel.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            **{k: v for k, v in kwargs.items() if k not in ["device", "torch_dtype", "task"]}
        )
        
        if device != "cpu":
            model = model.to(device)
            
        return model
    
    @staticmethod
    def synonyms_for_task(task: str) -> list:
        """Return synonyms for a given task."""
        # Reverse mapping of map_from_synonym
        reverse_synonyms = {
            "feature-extraction": ["sentence-similarity", "text-similarity"],
            "text-classification": ["zero-shot-classification"],
            "text-generation": ["causal-lm"],
            "text-generation-with-past": ["causal-lm-with-past"],
            "text2text-generation": ["seq2seq-lm"],
            "text2text-generation-with-past": ["seq2seq-lm-with-past"],
            "fill-mask": ["masked-lm"],
            "automatic-speech-recognition": ["asr", "speech-to-text"],
        }
        return reverse_synonyms.get(task, [])
    
    @staticmethod
    def get_exporter_config_constructor(exporter: str, model_type: str, task: str, **kwargs):
        """Get the config constructor for the given exporter, model type and task."""
        # This would normally return the appropriate config class
        # For now, return a dummy that creates a basic config
        from optimum.exporters.onnx.base import OnnxConfig
        return partial(OnnxConfig, task=task)
    
    @staticmethod
    def create_register(exporter: str, overwrite_existing: bool = False):
        """Create a register decorator for the given exporter."""
        def register(model_type: str, task: str):
            def decorator(config_class):
                # In a full implementation, this would register the config class
                return config_class
            return decorator
        return register


__all__ = ["TasksManager"]