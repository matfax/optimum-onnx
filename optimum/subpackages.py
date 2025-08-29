# Copyright 2022 The HuggingFace Team. All rights reserved.
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

"""Subpackages loading utilities."""

import importlib
import sys
from typing import Dict, Any

from .utils import is_onnxruntime_available


def load_subpackages():
    """Load available subpackages based on installed dependencies."""
    subpackages = {}
    
    if is_onnxruntime_available():
        try:
            subpackages['onnxruntime'] = importlib.import_module('optimum.onnxruntime')
        except ImportError:
            pass
    
    return subpackages


# Lazy loading for subpackages
_subpackages = None

def get_subpackages():
    global _subpackages
    if _subpackages is None:
        _subpackages = load_subpackages()
    return _subpackages