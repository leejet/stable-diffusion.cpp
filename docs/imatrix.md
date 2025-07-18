# Importance Matrix (imatrix) Quantization

## What is an Importance Matrix?

Quantization reduces the precision of a model's weights, decreasing its size and computational requirements. However, this can lead to a loss of quality. An importance matrix helps mitigate this by identifying which weights are *most* important for the model's performance. During quantization, these important weights are preserved with higher precision, while less important weights are quantized more aggressively.  This allows for better overall quality at a given quantization level.

This originates from work done with language models in [llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/examples/imatrix/README.md).

## Usage

The imatrix feature involves two main steps: *training* the matrix and *using* it during quantization.

### Training the Importance Matrix

To generate an imatrix, run stable-diffusion.cpp with the `--imat-out` flag, specifying the output filename.  This process runs alongside normal image generation.

```bash
sd.exe [same exact parameters as normal generation] --imat-out imatrix.dat
```

*   **`[same exact parameters as normal generation]`**:  Use the same command-line arguments you would normally use for image generation (e.g., prompt, dimensions, sampling method, etc.).
*   **`--imat-out imatrix.dat`**: Specifies the output file for the generated imatrix.

You can generate multiple images at once using the `-b` flag to speed up the training process.

### Continuing Training an Existing Matrix

If you want to refine an existing imatrix, use the `--imat-in` flag *in addition* to `--imat-out`. This will load the existing matrix and continue training it.

```bash
sd.exe [same exact parameters as normal generation] --imat-out imatrix.dat --imat-in imatrix.dat
```
With that, you can train and refine the imatrix while generating images like you'd normally do. 

### Using Multiple Matrices

You can load and merge multiple imatrices together:

```bash
sd.exe [same exact parameters as normal generation] --imat-out imatrix.dat --imat-in imatrix.dat --imat-in imatrix2.dat
```

### Quantizing with an Importance Matrix

To quantize a model using a trained imatrix, use the `-M convert` option (or equivalent quantization command) and the `--imat-in` flag, specifying the imatrix file.

```bash
sd.exe -M convert [same exact parameters as normal quantization] --imat-in imatrix.dat
```

*   **`[same exact parameters as normal quantization]`**: Use the same command-line arguments you would normally use for quantization (e.g., target quantization method, input/output filenames).
*   **`--imat-in imatrix.dat`**: Specifies the imatrix file to use during quantization.  You can specify multiple `--imat-in` flags to combine multiple matrices.

## Important Considerations

*   The quality of the imatrix depends on the prompts and settings used during training. Use prompts and settings representative of the types of images you intend to generate for the best results.
*   Experiment with different training parameters (e.g., number of images, prompt variations) to optimize the imatrix for your specific use case.
*   The performance impact of training an imatrix during image generation or using an imatrix for quantization is negligible.
*   Using already quantized models to train the imatrix seems to be working fine.