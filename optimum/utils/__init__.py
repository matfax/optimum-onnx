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

"""Minimal utilities for optimum."""

import logging

# Logging
def get_logger(name):
    return logging.getLogger(name)

# Versioning utilities
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

def is_diffusers_version(op, version):
    """Check diffusers version."""
    try:
        import diffusers
        from packaging import version as version_module
        current_version = version_module.Version(diffusers.__version__)
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

# Constants and defaults
class LoggingUtil:
    def get_logger(self, name):
        import logging
        return logging.getLogger(name)

logging = LoggingUtil()

DEFAULT_DUMMY_SHAPES = {
    "batch_size": 1,
    "sequence_length": 8,
    "height": 224,
    "width": 224,
    "num_channels": 3,
}

# Dummy classes for completeness
class DummyInputGenerator:
    pass

class DummyLabelsGenerator:
    pass

class DummySeq2SeqPastKeyValuesGenerator:
    pass

def is_diffusers_available():
    try:
        import diffusers
        return True
    except ImportError:
        return False