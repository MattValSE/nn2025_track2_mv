import torch
import torch.onnx
import numpy as np
from new_model import GRUNetwork


def convert_pytorch_to_onnx(pytorch_model_path, onnx_output_path, 
                           input_shape=(1, 1, 642), hidden_size=322,
                           model_class=None):
    """
    Convert PyTorch model to ONNX format
    
    Args:
        pytorch_model_path: Path to PyTorch model (.pth/.pt file)
        onnx_output_path: Output path for ONNX model
        input_shape: Shape of input features (batch, seq, features)
        hidden_size: Size of hidden states
        model_class: Model class if loading state dict only
    """
    
    # Load PyTorch model
    device = torch.device('cpu')  # Use CPU for conversion
    
    if model_class is not None:
        # If you need to instantiate the model class first
        model = model_class()
        model.load_state_dict(torch.load(pytorch_model_path, map_location=device))
    else:
        # If the model was saved with torch.save(model, path)
        model = torch.load(pytorch_model_path, map_location=device)
    
    model.eval()
    
    # Create dummy inputs matching the expected format
    dummy_input = torch.randn(input_shape, dtype=torch.float32)
    dummy_h01 = torch.zeros((1, 1, hidden_size), dtype=torch.float32)
    dummy_h02 = torch.zeros((1, 1, hidden_size), dtype=torch.float32)
    
    # Define input names (must match what the original script expects)
    input_names = ['input', 'h01', 'h02']
    output_names = ['output','hn1','hn2']  # Adjust based on your model
    
    # Dynamic axes for variable sequence length (optional)
    dynamic_axes = {
        'input': {1: 'sequence_length'},
        'output': {1: 'sequence_length'}
    }
    
    print(f"Converting model to ONNX...")
    print(f"Input shape: {input_shape}")
    print(f"Hidden size: {hidden_size}")
    
    try:
        with torch.no_grad():
            torch.onnx.export(
                model,
                (dummy_input, dummy_h01, dummy_h02),  # Tuple of inputs
                onnx_output_path,
                export_params=True,
                opset_version=11,  # Use opset 11 for better compatibility
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes
            )
        
        print(f"Successfully converted to ONNX: {onnx_output_path}")
        
        # Verify the conversion
        verify_onnx_model(onnx_output_path, dummy_input, dummy_h01, dummy_h02)
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        print("Common issues and solutions:")
        print("1. Check if your model supports ONNX export")
        print("2. Verify input shapes match your model's expectations")
        print("3. Some PyTorch operations may not be supported in ONNX")

def verify_onnx_model(onnx_path, dummy_input, dummy_h01, dummy_h02):
    """Verify the ONNX model works correctly"""
    try:
        import onnxruntime
        
        # Load ONNX model
        session = onnxruntime.InferenceSession(onnx_path)
        
        # Run inference
        inputs = {
            'input': dummy_input.numpy(),
            'h01': dummy_h01.numpy(),
            'h02': dummy_h02.numpy()
        }
        
        outputs = session.run(None, inputs)
        print(f"ONNX model verification successful!")
        print(f"Output shapes: {[out.shape for out in outputs]}")
        
    except ImportError:
        print("onnxruntime not installed. Install with: pip install onnxruntime")
    except Exception as e:
        print(f"ONNX verification failed: {e}")

# Example usage for different model types
# Simple conversion function for your specific case
def convert_best_model():
    """Convert your best_model.pt to ONNX format"""
    state_dict = torch.load("best_model.pt", map_location='cpu')
    # Load your model
    model = GRUNetwork()  
    model.load_state_dict(state_dict)
    model.eval()
    
    # Create dummy inputs - YOU NEED TO ADJUST THESE SHAPES!
    # Based on the original script, it expects:
    dummy_input = torch.randn(1, 1, 322, dtype=torch.float32)  # Features: [batch, seq, features]
    dummy_h01 = torch.zeros(1, 1, 322, dtype=torch.float32)    # Hidden state 1
    dummy_h02 = torch.zeros(1, 1, 322, dtype=torch.float32)    # Hidden state 2
    
    # Input and output names (must match original script expectations)
    input_names = ['input', 'h01', 'h02']
    output_names = ['output','hn1','hn2']  # Adjust based on your model outputs
    
    print("Converting best_model.pt to ONNX...")
    print(f"Input shapes: {dummy_input.shape}, {dummy_h01.shape}, {dummy_h02.shape}")
    
    try:
        # Test your model first to see what it outputs
        with torch.no_grad():
            test_output = model(dummy_input, dummy_h01, dummy_h02)
            print(f"Model output type: {type(test_output)}")
            if isinstance(test_output, tuple):
                print(f"Output shapes: {[out.shape for out in test_output]}")
            else:
                print(f"Single output shape: {test_output.shape}")
        
        # Export to ONNX
        torch.onnx.export(
            model,
            (dummy_input, dummy_h01, dummy_h02),  # Tuple of dummy inputs
            "best_model_converted.onnx",
            input_names=input_names,
            output_names=output_names,
            opset_version=11,  # Use 11 for better compatibility
            verbose=True,
            do_constant_folding=True
        )
        
        print("✓ Successfully converted to best_model_converted.onnx")
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        print("\nTroubleshooting steps:")
        print("1. Check if your model expects different input shapes")
        print("2. Verify your model's forward() method signature")
        print("3. Make sure all operations in your model are ONNX-compatible")

if __name__ == "__main__":
    # Convert your specific model
    convert_best_model()
    
    # Alternative: Use the general function
    """
    convert_pytorch_to_onnx(
        pytorch_model_path="best_model.pt",
        onnx_output_path="best_model_converted.onnx",
        input_shape=(1, 1, 322),  # ADJUST THESE BASED ON YOUR MODEL!
        hidden_size=322
    )
    """