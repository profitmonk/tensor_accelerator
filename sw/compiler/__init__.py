"""
Tensor Accelerator Compiler

Compiles ONNX models to accelerator assembly code.
"""

from .compile import Compiler, compile_model

__all__ = ['Compiler', 'compile_model']
