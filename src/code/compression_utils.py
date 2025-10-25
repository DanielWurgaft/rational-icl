import numpy as np
import torch
import lzma
import bz2
import brotli
import os
import ast
import tokenize
import io
import torch
from pyprojroot import here
import zstandard as zstd

def compress_array(data):
    """
    Compresses a NumPy array or PyTorch tensor using multiple high-performance
    compression algorithms and returns the smallest size achieved.
    Preserves the original data type.

    Parameters:
    - data (np.ndarray or torch.Tensor): The input array or tensor to compress.

    Returns:
    - int: The size of the compressed data in bits using the best algorithm.
    - str: Name of the algorithm that produced the smallest size.
    """
    # Convert PyTorch tensor to NumPy array if necessary
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    if isinstance(data, list):
        data = np.array(data)
    elif not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a NumPy array, list, or PyTorch tensor.")

    # Serialize the data to bytes (keeping original data type)
    data_bytes = data.tobytes()
    
    # Store original size as a baseline option
    original_size = len(data_bytes) * 8
    compression_results = {'none': original_size}

    # Try different compression algorithms
    
    # LZMA compression - highest setting (typically good for binary data)
    try:
        lzma_compressed = lzma.compress(data_bytes, preset=9 | lzma.PRESET_EXTREME)
        compression_results['lzma'] = len(lzma_compressed) * 8
    except Exception as e:
        print(f"LZMA compression failed: {e}")
    
    # BZIP2 compression - highest setting
    try:
        bzip2_compressed = bz2.compress(data_bytes, compresslevel=9)
        compression_results['bzip2'] = len(bzip2_compressed) * 8
    except Exception as e:
        print(f"BZIP2 compression failed: {e}")
        
    # Brotli compression - highest quality setting
    try:
        brotli_compressed = brotli.compress(data_bytes, quality=11, mode=brotli.MODE_GENERIC)
        compression_results['brotli'] = len(brotli_compressed) * 8
    except Exception as e:
        print(f"Brotli compression failed: {e}")
    
    # Zstandard compression (excellent for numeric data)
    try:
        # Level 22 is the highest compression level for zstd
        zstd_compressor = zstd.ZstdCompressor(level=22)
        zstd_compressed = zstd_compressor.compress(data_bytes)
        compression_results['zstd'] = len(zstd_compressed) * 8
    except Exception as e:
        print(f"Zstandard compression failed: {e}")

    # Try delta encoding + compression for numeric arrays
    if data.dtype.kind in 'iufc':  # Integer, unsigned integer, float, complex
        try:
            # Delta encode the data (difference between consecutive elements)
            delta_data = np.diff(data, prepend=data.flat[0])
            delta_bytes = delta_data.tobytes()
            
            # Compress the delta-encoded data with Brotli
            delta_compressed = brotli.compress(delta_bytes, quality=11, mode=brotli.MODE_GENERIC)
            compression_results['delta+brotli'] = len(delta_compressed) * 8
        except Exception as e:
            print(f"Delta+Brotli compression failed: {e}")

    # Find the algorithm with the smallest size
    best_algorithm = min(compression_results, key=compression_results.get)
    best_size = compression_results[best_algorithm]
    
    return best_size

def compress_python_script(class_name, file_path='src/code/models.py'):
    """
    Compresses a Python script file using multiple high-performance compression 
    algorithms and returns the smallest size achieved.

    Parameters:
    - class_name (str): Name of the class to extract and compress.
    - file_path (str): Path to the Python script file.

    Returns:
    - int: Size of the compressed data in bits using the best algorithm.
    - str: Name of the algorithm that produced the smallest size.
    """
    # Read the script file
    script_str = get_class_source_code(class_name, here(file_path))
    # Convert to bytes
    script_bytes = script_str.encode('utf-8')
    
    # Store original size as a baseline option
    original_size = len(script_bytes) * 8
    compression_results = {'none': original_size}

    # Try different compression algorithms
    
    # LZMA compression - highest setting
    try:
        lzma_compressed = lzma.compress(script_bytes, preset=9 | lzma.PRESET_EXTREME)
        compression_results['lzma'] = len(lzma_compressed) * 8
    except Exception as e:
        print(f"LZMA compression failed: {e}")
    
    # BZIP2 compression - highest setting
    try:
        bzip2_compressed = bz2.compress(script_bytes, compresslevel=9)
        compression_results['bzip2'] = len(bzip2_compressed) * 8
    except Exception as e:
        print(f"BZIP2 compression failed: {e}")
        
    # Brotli compression - highest quality setting (excellent for text)
    try:
        brotli_compressed = brotli.compress(script_bytes, quality=11, mode=brotli.MODE_TEXT)
        compression_results['brotli'] = len(brotli_compressed) * 8
    except Exception as e:
        print(f"Brotli compression failed: {e}")
    
    # Zstandard compression (excellent for code)
    try:
        # Use dictionary compression mode which is ideal for code
        zstd_compressor = zstd.ZstdCompressor(level=22)
        zstd_compressed = zstd_compressor.compress(script_bytes)
        compression_results['zstd'] = len(zstd_compressed) * 8
    except Exception as e:
        print(f"Zstandard compression failed: {e}")

    # Find the algorithm with the smallest size
    best_algorithm = min(compression_results, key=compression_results.get)
    best_size = compression_results[best_algorithm]
    
    return best_size

def get_class_source_code(class_name, file_path):
    """
    Reads the specified Python file, extracts the source code of the given class,
    and removes all comments, spaces, and documentation.

    Args:
        class_name (str): The name of the class to extract.
        file_path (str): The path to the Python file containing the class.

    Returns:
        str: The cleaned source code of the specified class.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the class with the given name is not found.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    with open(file_path, 'r') as file:
        source = file.read()

    # Parse the source code into an AST
    tree = ast.parse(source)

    class_node = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            class_node = node
            break

    if not class_node:
        raise ValueError(f"Class {class_name} not found in {file_path}.")

    # Remove docstrings by creating a new class node without the docstring
    if (len(class_node.body) > 0 and isinstance(class_node.body[0], ast.Expr) and 
        isinstance(class_node.body[0].value, ast.Str)):
        del class_node.body[0]

    # Convert the class node back to source code
    class_code = ast.unparse(class_node)

    # Remove all comments and unnecessary whitespace
    def remove_comments_and_whitespace(code):
        io_obj = io.StringIO(code)
        out = ""
        prev_toktype = tokenize.INDENT
        last_lineno = -1
        last_col = 0
        try:
            for tok in tokenize.generate_tokens(io_obj.readline):
                token_type = tok.type
                token_string = tok.string
                start_line, start_col = tok.start
                end_line, end_col = tok.end
                if token_type == tokenize.COMMENT:
                    continue
                elif token_type == tokenize.STRING and prev_toktype == tokenize.INDENT:
                    # Skip docstrings
                    continue
                elif token_type in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT):
                    continue
                else:
                    out += token_string
                prev_toktype = token_type
        except tokenize.TokenError:
            pass
        return ''.join(out.split())

    cleaned_code = remove_comments_and_whitespace(class_code)

    return cleaned_code