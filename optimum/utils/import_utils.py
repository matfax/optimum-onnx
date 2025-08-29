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

"""Import utilities."""

def is_torch_available():
    try:
        import torch
        return True
    except ImportError:
        return False

def is_onnx_available():
    try:
        import onnx
        return True
    except ImportError:
        return False

def is_onnxruntime_available():
    try:
        import onnxruntime
        return True
    except ImportError:
        return False

def is_transformers_version(op, version):
    """Check transformers version."""
    try:
        import transformers
        from packaging import version as version_module
        current_version = version_module.Version(transformers.__version__)
        target_version = version_module.Version(version)
        
        if op == ">=":
            return current_version >= target_version
        elif op == ">":
            return current_version > target_version
        elif op == "<=":
            return current_version <= target_version
        elif op == "<":
            return current_version < target_version
        elif op == "==":
            return current_version == target_version
        else:
            raise ValueError(f"Unsupported operator: {op}")
    except ImportError:
        return False