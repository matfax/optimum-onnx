#!/usr/bin/env python3

"""
Example of how to use the Int32Wrapper during ONNX export.

This example shows how to enable int32 input conversion during ONNX export,
which can improve compatibility with certain ONNX runtimes and hardware accelerators.
"""

import tempfile
from pathlib import Path
import torch
from transformers import GPT2LMHeadModel, GPT2Config
import sys
import os

# Add the local optimum package to the path
sys.path.insert(0, '/home/runner/work/optimum-onnx/optimum-onnx')

from optimum.exporters.onnx.base import OnnxConfig
from optimum.exporters.onnx.convert import export_pytorch
import onnx
import numpy as np
from onnxruntime import InferenceSession


class GPT2OnnxConfig(OnnxConfig):
    """Example ONNX configuration for GPT2 with int32 input support."""
    
    def __init__(self, config, task="text-generation", use_int32_inputs=False):
        super().__init__(
            config=config, 
            task=task,
            use_int32_inputs=use_int32_inputs  # Enable int32 wrapper
        )
    
    @property
    def inputs(self):
        return {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
        }
    
    @property
    def outputs(self):
        return {
            "logits": {0: "batch_size", 1: "sequence_length"},
        }
    
    def generate_dummy_inputs(self, framework="pt", **kwargs):
        dummy_input = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        return {"input_ids": dummy_input}


def main():
    print("Example: Using Int32Wrapper during ONNX export")
    print("=" * 50)
    
    # Create a simple test model
    config = GPT2Config(
        vocab_size=100,
        n_positions=8,
        n_embd=32,
        n_layer=1,
        n_head=2,
    )
    model = GPT2LMHeadModel(config)
    model.eval()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        
        # Export without int32 wrapper (default behavior)
        print("\n1. Exporting model WITHOUT int32 wrapper...")
        output_path_normal = Path(temp_dir) / "model_normal.onnx"
        
        onnx_config_normal = GPT2OnnxConfig(
            config=model.config,
            use_int32_inputs=False  # Default behavior
        )
        
        export_pytorch(
            model=model,
            config=onnx_config_normal,
            opset=14,
            output=output_path_normal,
            device="cpu"
        )
        print(f"✓ Exported to {output_path_normal}")
        
        # Export with int32 wrapper enabled
        print("\n2. Exporting model WITH int32 wrapper...")
        output_path_int32 = Path(temp_dir) / "model_int32.onnx"
        
        onnx_config_int32 = GPT2OnnxConfig(
            config=model.config,
            use_int32_inputs=True  # Enable int32 wrapper
        )
        
        export_pytorch(
            model=model,
            config=onnx_config_int32,
            opset=14,
            output=output_path_int32,
            device="cpu"
        )
        print(f"✓ Exported to {output_path_int32}")
        
        # Test both models
        print("\n3. Testing both exported models...")
        
        # Test normal model
        session_normal = InferenceSession(str(output_path_normal), providers=["CPUExecutionProvider"])
        test_input_int64 = np.array([[1, 2, 3, 4]], dtype=np.int64)
        outputs_normal = session_normal.run(None, {"input_ids": test_input_int64})
        print("✓ Normal model works with int64 inputs")
        
        # Test int32 model 
        session_int32 = InferenceSession(str(output_path_int32), providers=["CPUExecutionProvider"])
        
        # Should work with int64 inputs (gets converted to int32 internally)
        outputs_int32_from_int64 = session_int32.run(None, {"input_ids": test_input_int64})
        print("✓ Int32 model works with int64 inputs")
        
        # Should also work with int32 inputs directly
        test_input_int32 = np.array([[1, 2, 3, 4]], dtype=np.int32)
        outputs_int32_from_int32 = session_int32.run(None, {"input_ids": test_input_int32})
        print("✓ Int32 model works with int32 inputs")
        
        # Compare outputs - they should be very similar
        diff = np.max(np.abs(outputs_normal[0] - outputs_int32_from_int64[0]))
        print(f"✓ Max difference between normal and int32 models: {diff:.6f}")
        
        print("\n4. Summary:")
        print("  - The int32 wrapper transparently converts int64 input_ids to int32")
        print("  - Both models produce very similar results")
        print("  - The int32 model can work with both int64 and int32 inputs")
        print("  - This can improve compatibility with certain ONNX runtimes")


if __name__ == "__main__":
    main()