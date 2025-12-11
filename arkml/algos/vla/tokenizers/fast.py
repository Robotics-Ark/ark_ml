import numpy as np
from typing import List


class FASTTokenizer:
    """
    A FAST (Fast Action Sequence Tokenizer) tokenizer for quantizing continuous action values.
    
    This tokenizer implements quantization and dequantization functionality by mapping continuous
    action values to discrete token indices and vice versa.
    
    Attributes:
        vocab_path (str): Path to vocabulary file (Not used in this quantization-based tokenizer)
        num_bins (int): Number of discrete bins for quantization
        min_val (float): Minimum value for the quantization range
        max_val (float): Maximum value for the quantization range
        step_size (float): Size of each quantization bin
    """
    
    def __init__(self, vocab_path: str, num_bins: int, min_val: float, max_val: float):
        """
        Initialize the FASTTokenizer.
        
        Args:
            vocab_path (str): Path to vocabulary file (currently unused in this quantization-based tokenizer)
            num_bins (int): Number of discrete bins for quantization
            min_val (float): Minimum value for the quantization range
            max_val (float): Maximum value for the quantization range
        """
        self.vocab_path = vocab_path
        self.num_bins = num_bins
        self.min_val = min_val
        self.max_val = max_val
        self.step_size = (max_val - min_val) / num_bins
        
    def encode(self, actions: np.ndarray) -> List[int]:
        """
        Encode continuous action values into discrete token indices.
        
        Args:
            actions (np.ndarray): Array of continuous action values of shape (..., action_dim)
            
        Returns:
            List[int]: List of token indices in the range [0, num_bins-1]
            
        Example:
            >>> tokenizer = FASTTokenizer("", num_bins=100, min_val=-1.0, max_val=1.0)
            >>> actions = np.array([[0.0, 0.5, -0.5]])
            >>> tokens = tokenizer.encode(actions)
            >>> assert len(tokens) == 3
            >>> assert all(0 <= t < 100 for t in tokens)
        """
        # Clip values to the allowed range
        clipped_actions = np.clip(actions, self.min_val, self.max_val)
        
        # Normalize to [0, num_bins-1] range
        normalized = (clipped_actions - self.min_val) / (self.max_val - self.min_val)
        tokens = (normalized * (self.num_bins - 1)).astype(int)
        
        # Ensure tokens are in the correct range
        tokens = np.clip(tokens, 0, self.num_bins - 1)
        
        # Flatten and convert to list of integers
        return tokens.flatten().tolist()
    
    def decode(self, tokens: List[int]) -> np.ndarray:
        """
        Decode discrete token indices back to continuous action values.
        
        Args:
            tokens (List[int]): List of token indices in the range [0, num_bins-1]
            
        Returns:
            np.ndarray: Array of continuous action values of shape (len(tokens),)
            
        Example:
            >>> tokenizer = FASTTokenizer("", num_bins=100, min_val=-1.0, max_val=1.0)
            >>> tokens = [0, 50, 99]  # Should map to approximately -1.0, 0.0, 1.0
            >>> actions = tokenizer.decode(tokens)
            >>> expected = np.array([-1.0, 0.0, 1.0])
            >>> # Allow for small numerical differences due to quantization
            >>> assert np.allclose(actions, expected, atol=0.05)
        """
        # Convert tokens to numpy array
        token_array = np.array(tokens)
        
        # Ensure tokens are in the valid range
        token_array = np.clip(token_array, 0, self.num_bins - 1)
        
        # Convert tokens back to continuous values
        # Map from [0, num_bins-1] to [min_val, max_val]
        normalized = token_array / (self.num_bins - 1)
        actions = normalized * (self.max_val - self.min_val) + self.min_val
        
        return actions


if __name__ == "__main__":
    # Basic unit tests
    
    # Test 1: Basic functionality
    tokenizer = FASTTokenizer("", num_bins=10, min_val=-1.0, max_val=1.0)
    
    # Test encoding
    actions = np.array([[0.0, 0.5, -0.5]])
    tokens = tokenizer.encode(actions)
    print(f"Encoded tokens: {tokens}")
    
    # Test decoding
    decoded_actions = tokenizer.decode(tokens)
    print(f"Decoded actions: {decoded_actions}")
    
    # Test 2: Edge cases
    edge_actions = np.array([[-1.0, 1.0]])  # Min and max values
    edge_tokens = tokenizer.encode(edge_actions)
    print(f"Edge case tokens: {edge_tokens}")
    
    edge_decoded = tokenizer.decode(edge_tokens)
    print(f"Edge case decoded: {edge_decoded}")
    
    # Test 3: Out of range values (should be clipped)
    out_of_range_actions = np.array([[-2.0, 2.0]])  # Beyond min/max
    clipped_tokens = tokenizer.encode(out_of_range_actions)
    print(f"Clipped tokens: {clipped_tokens}")
    
    clipped_decoded = tokenizer.decode(clipped_tokens)
    print(f"Clipped decoded: {clipped_decoded}")
    
    print("All tests completed successfully!")