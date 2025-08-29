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

"""Exporters package for optimum-onnx."""

# Try to import TasksManager from the main optimum package if available
try:
    from optimum.exporters.tasks import TasksManager
except ImportError:
    # If not available, we'll need to provide a minimal implementation
    # This is a placeholder - in production, this should come from the main optimum package
    class TasksManager:
        """Minimal TasksManager implementation for development purposes."""
        
        @staticmethod
        def get_all_tasks():
            return ["feature-extraction", "text-classification", "token-classification", 
                   "text-generation", "text2text-generation", "fill-mask"]
        
        @staticmethod
        def map_from_synonym(task):
            return task
        
        @staticmethod
        def determine_framework(*args, **kwargs):
            return "pt"
        
        @staticmethod
        def infer_library_from_model(*args, **kwargs):
            return "transformers"
        
        @staticmethod
        def infer_task_from_model(*args, **kwargs):
            return "feature-extraction"
        
        @staticmethod
        def get_supported_tasks_for_model_type(*args, **kwargs):
            return ["feature-extraction"]
        
        @staticmethod
        def get_model_from_task(*args, **kwargs):
            from transformers import AutoModel
            return AutoModel.from_pretrained(args[1])
        
        @staticmethod
        def synonyms_for_task(task):
            return []
        
        _SUPPORTED_MODEL_TYPE = {}

# Make TasksManager available at the package level
__all__ = ["TasksManager"]