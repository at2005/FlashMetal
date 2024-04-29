# FlashMetal
FlashAttention for Apple Silicon

FlashAttention  working on Apple Silicon as a custom kernel MPS extension.

When using the .so file, move it to your project directory alongside .metallib precompiled shader.

If building locally then have Metal-C++ directory in home, or change the path in buildscript.

You can wrap it in a custom autograd function and use in PyTorch.

Optimisations incoming!
