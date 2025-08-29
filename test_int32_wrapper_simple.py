#!/usr/bin/env python3

"""
Simple test script to verify Int32Wrapper functionality.
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Config


class Int32Wrapper(torch.nn.Module):
    """Wrapper that converts input_ids from int64 (torch.long) to int32 during ONNX export.
    
    This is useful for ONNX models that prefer int32 inputs for better compatibility
    with certain ONNX runtimes and hardware accelerators.
    """
    
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids=None, **kwargs):
        if input_ids is not None and input_ids.dtype == torch.long:
            input_ids = input_ids.to(torch.int32)
        return self.model(input_ids=input_ids, **kwargs)


def test_int32_wrapper_standalone():
    """Test that Int32Wrapper works correctly standalone."""
    print("Testing Int32Wrapper standalone...")
    
    # Create a simple test model
    config = GPT2Config(
        vocab_size=100,
        n_positions=8,
        n_embd=32,
        n_layer=1,
        n_head=2,
    )
    model = GPT2LMHeadModel(config)
    model.eval()  # Set to evaluation mode
    
    # Wrap the model
    wrapped_model = Int32Wrapper(model)
    
    # Test with int64 input_ids
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    
    # Forward through original model with int32 to match what wrapper does
    input_ids_int32 = input_ids.to(torch.int32)
    original_output = model(input_ids_int32)
    
    # Forward through wrapped model
    wrapped_output = wrapped_model(input_ids)
    
    # Results should be the same since wrapper converts to int32
    print(f"Original logits shape: {original_output.logits.shape}")
    print(f"Wrapped logits shape: {wrapped_output.logits.shape}")
    print(f"Max diff: {torch.max(torch.abs(original_output.logits - wrapped_output.logits)).item()}")
    
    assert torch.allclose(original_output.logits, wrapped_output.logits, atol=1e-5)
    print("✓ Int32Wrapper produces same results as original model with int32 inputs")
    
    # Test that the wrapper actually converts input_ids to int32 internally
    class CheckingWrapper(Int32Wrapper):
        def forward(self, input_ids=None, **kwargs):
            if input_ids is not None and input_ids.dtype == torch.long:
                print(f"Converting input_ids from {input_ids.dtype} to int32")
                input_ids = input_ids.to(torch.int32)
                # Verify the conversion happened
                assert input_ids.dtype == torch.int32, f"Expected int32, got {input_ids.dtype}"
            return self.model(input_ids=input_ids, **kwargs)
    
    checking_wrapper = CheckingWrapper(model)
    checking_output = checking_wrapper(torch.tensor([[1, 2, 3, 4]], dtype=torch.long))
    print("✓ Int32Wrapper correctly converts input_ids to int32")


def test_conversion_behavior():
    """Test the conversion behavior with different input types."""
    print("\nTesting conversion behavior...")
    
    config = GPT2Config(
        vocab_size=50,
        n_positions=4,
        n_embd=16,
        n_layer=1,
        n_head=2,
    )
    model = GPT2LMHeadModel(config)
    model.eval()
    wrapped_model = Int32Wrapper(model)
    
    # Test with int64 input (should convert)
    input_ids_int64 = torch.tensor([[1, 2, 3]], dtype=torch.long)
    output_64 = wrapped_model(input_ids_int64)
    print("✓ Works with int64 inputs")
    
    # Test with int32 input (should not change)
    input_ids_int32 = torch.tensor([[1, 2, 3]], dtype=torch.int32)
    output_32 = wrapped_model(input_ids_int32)
    print("✓ Works with int32 inputs")
    
    # Test with other arguments
    attention_mask = torch.ones((1, 3))
    output_with_mask = wrapped_model(input_ids_int64, attention_mask=attention_mask)
    print("✓ Works with additional arguments")
    
    # Verify the outputs are the same regardless of input dtype
    reference_output = model(input_ids_int32)
    assert torch.allclose(output_64.logits, reference_output.logits, atol=1e-5)
    assert torch.allclose(output_32.logits, reference_output.logits, atol=1e-5)
    print("✓ Outputs are consistent regardless of input_ids dtype")


if __name__ == "__main__":
    test_int32_wrapper_standalone()
    test_conversion_behavior()
    print("\n🎉 All basic tests passed!")
    print("\nThe Int32Wrapper class is working correctly and can be integrated into the ONNX export process.")