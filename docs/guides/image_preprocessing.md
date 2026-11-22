# Image preprocessing

Custom input transformations are expressed in [DGL](../dgl.md). Native CLI and
C API generation use the model's default canvas and reference-image rules.
The former `--image-preprocess` flag, server `image_preprocess` field and C API
`image_preprocess.rules` option have been removed. Express their transformations
in DGL instead. `--ref-image-args` and `ref_image_args` have also been removed.

Use tensor operations to make the transformations visible in the program:

```dgl
image = image::load("input.png")
cropped = tensor::crop(image, start = [32, 16], size = [512, 384], axes = [1, 2])
resized = tensor::interpolate(cropped, size = [768, 512], axes = [1, 2], mode = "lanczos")
pixels = image::to_f32(resized)
```

For a mask, use `mode = "nearest_exact"` when resizing and explicitly convert to
the expected mask layout. For an edge hint, use `image::canny(image)` and pass
the result to the ControlNet workflow. `tensor::pad`, `tensor::crop` and
`tensor::interpolate` can express fitting, padding and aspect-ratio handling.

These operations do not disable mandatory VAE or vision-encoder constraints.
Prepare an input that already satisfies those constraints to avoid automatic
alignment. Different encoders may receive different prepared tensors.

See the [editing guide](edit.md), the
[ControlNet example](../dgl/sd/sd1-5-controlnet.dgl), and the
[built-in catalog](../dgl-builtins.md) for complete interfaces.
