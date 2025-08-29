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

"""Base classes for model exporters."""

from abc import ABC, abstractmethod
from typing import Any


class ExporterConfig(ABC):
    """Base class for exporter configurations."""
    
    def __init__(self, config, task: str = "feature-extraction", int_dtype: str = "int64", float_dtype: str = "fp32"):
        self.config = config
        self.task = task
        self.int_dtype = int_dtype
        self.float_dtype = float_dtype
    
    @property
    @abstractmethod
    def inputs(self):
        """The inputs of the model."""
        pass
    
    @property
    @abstractmethod
    def outputs(self):
        """The outputs of the model."""
        pass
    
    @abstractmethod
    def generate_dummy_inputs(self, framework: str = "pt", **kwargs):
        """Generate dummy inputs for the model."""
        pass