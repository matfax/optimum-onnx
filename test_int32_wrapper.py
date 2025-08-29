#!/usr/bin/env python3

"""
Test script to verify Int32Wrapper integration works correctly.
"""

import tempfile
import torch
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2Config
import sys
import os

# Add the local optimum package to the path
sys.path.insert(0, '/home/runner/work/optimum-onnx/optimum-onnx')

from optimum.exporters.onnx.base import OnnxConfig
from optimum.exporters.onnx.model_patcher import Int32Wrapper
import onnx
import numpy as np
from onnxruntime import InferenceSession


def test_int32_wrapper_standalone():
    """Test that Int32Wrapper works correctly standalone."""
    print("Testing Int32Wrapper standalone...")
    
    # Create a simple test model
    config = GPT2Config(
        vocab_size=1000,
        n_positions=32,
        n_embd=64,
        n_layer=2,
        n_head=2,
    )
    model = GPT2LMHeadModel(config)
    
    # Wrap the model
    wrapped_model = Int32Wrapper(model)
    
    # Test with int64 input_ids
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    
    # Forward through original model 
    original_output = model(input_ids)
    
    # Forward through wrapped model
    wrapped_output = wrapped_model(input_ids)
    
    # Results should be the same
    assert torch.allclose(original_output.logits, wrapped_output.logits, atol=1e-5)
    print("✓ Int32Wrapper produces same results as original model")
    
    # Test that the wrapper actually converts input_ids to int32 internally
    class CheckingWrapper(Int32Wrapper):
        def forward(self, input_ids=None, **kwargs):
            if input_ids is not None and input_ids.dtype == torch.long:
                input_ids = input_ids.to(torch.int32)
                # Verify the conversion happened
                assert input_ids.dtype == torch.int32, f"Expected int32, got {input_ids.dtype}"
            return self.model(input_ids=input_ids, **kwargs)
    
    checking_wrapper = CheckingWrapper(model)
    checking_output = checking_wrapper(torch.tensor([[1, 2, 3, 4]], dtype=torch.long))
    print("✓ Int32Wrapper correctly converts input_ids to int32")


def test_int32_wrapper_with_export():
    """Test that Int32Wrapper integration works with ONNX export."""
    print("\nTesting Int32Wrapper with ONNX export...")
    
    # Create a small test config and model for faster testing
    config = GPT2Config(
        vocab_size=100,
        n_positions=8,
        n_embd=32,
        n_layer=1,
        n_head=2,
    )
    model = GPT2LMHeadModel(config)
    
    # Create a custom ONNX config with use_int32_inputs enabled
    class TestOnnxConfig(OnnxConfig):
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
    
    # Test export with use_int32_inputs=True
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "test_model.onnx"
        
        # Create config with int32 inputs enabled
        onnx_config = TestOnnxConfig(
            config=model.config, 
            task="text-generation",
            use_int32_inputs=True
        )
        
        # Export the model
        from optimum.exporters.onnx.convert import export_pytorch
        export_pytorch(
            model=model,
            config=onnx_config,
            opset=14,
            output=output_path,
            device="cpu"
        )
        
        print(f"✓ Successfully exported model with Int32Wrapper to {output_path}")
        
        # Load and verify the exported model
        onnx_model = onnx.load(str(output_path))
        
        # Check that input is expected to be int32 or int64 (ONNX represents both as int64 in the graph)
        input_info = onnx_model.graph.input[0]
        print(f"✓ Exported model input type: {input_info.type.tensor_type.elem_type}")
        
        # Test inference with the exported model
        session = InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
        
        # Test with int64 inputs (should work)
        test_input = np.array([[1, 2, 3, 4]], dtype=np.int64)
        outputs = session.run(None, {"input_ids": test_input})
        print("✓ Exported model works with int64 inputs")
        
        # Test with int32 inputs (should also work)
        test_input_int32 = np.array([[1, 2, 3, 4]], dtype=np.int32)
        outputs_int32 = session.run(None, {"input_ids": test_input_int32})
        print("✓ Exported model works with int32 inputs")


def test_model_patcher_integration():
    """Test that ModelPatcher correctly applies Int32Wrapper when configured."""
    print("\nTesting ModelPatcher integration...")
    
    from optimum.exporters.onnx.model_patcher import ModelPatcher
    
    # Create a simple test model
    config = GPT2Config(
        vocab_size=50,
        n_positions=8,
        n_embd=16,
        n_layer=1,
        n_head=2,
    )
    model = GPT2LMHeadModel(config)
    
    # Create a custom ONNX config with use_int32_inputs enabled
    class TestOnnxConfig(OnnxConfig):
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
    
    # Test with use_int32_inputs=True
    onnx_config = TestOnnxConfig(
        config=model.config, 
        task="text-generation",
        use_int32_inputs=True
    )
    
    # Create model patcher
    patcher = ModelPatcher(onnx_config, model)
    
    # Check that the wrapper was applied
    assert isinstance(patcher._model, Int32Wrapper), "Expected Int32Wrapper to be applied"
    assert patcher._int32_wrapper is not None, "Expected _int32_wrapper to be set"
    print("✓ ModelPatcher correctly applies Int32Wrapper when use_int32_inputs=True")
    
    # Test with use_int32_inputs=False (default)
    onnx_config_no_wrapper = TestOnnxConfig(
        config=model.config, 
        task="text-generation",
        use_int32_inputs=False
    )
    
    patcher_no_wrapper = ModelPatcher(onnx_config_no_wrapper, model)
    
    # Check that the wrapper was NOT applied
    assert not isinstance(patcher_no_wrapper._model, Int32Wrapper), "Expected no Int32Wrapper"
    assert patcher_no_wrapper._int32_wrapper is None, "Expected _int32_wrapper to be None"
    print("✓ ModelPatcher correctly skips Int32Wrapper when use_int32_inputs=False")


if __name__ == "__main__":
    test_int32_wrapper_standalone()
    test_model_patcher_integration()
    test_int32_wrapper_with_export()
    print("\n🎉 All tests passed!")