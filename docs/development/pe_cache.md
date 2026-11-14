# Position embedding cache

`sd::PECache` stores CPU-side `sd::Tensor<float>` data. Each model runner owns
its cache and calls the existing `make_input()` on every graph build. Cache
hits skip PE generation; allocation and upload remain ordinary graph-input
operations. `GGMLRunner` has no PE-specific code or resource hooks.

The cache keeps the most recent key and tensor for each named PE input. A
matching key returns the existing tensor by reference. Changing its parameters
replaces that input's cached result. Different named inputs have independent
storage. Alternating two layouts under the same name recomputes their PE;
there is no LRU, device-buffer cache, memory budget, or graph lifecycle callback.
CPU data lives with the model runner, including across `runner_end()` calls,
and is released when its named entry is replaced or the cache is destroyed.

## Adding a model

Include `core/pe_cache.h` and declare a member:

```cpp
sd::PECache pe_cache_;
```

Construct the key before generating PE and place the generation code in the
lazy callback:

```cpp
sd::PECacheKey key("model.image_pe");
key.add(height).add(width).add(head_dim).add(theta)
    .add(circular_x_enabled).add(circular_y_enabled);
auto pe = make_input(pe_cache_.get_or_create(key, [&]() {
    auto values = generate_position_embedding(/* existing arguments */);
    return sd::Tensor<float>({2, 2, head_dim / 2, token_count}, std::move(values));
}));
```

Keys must cover every variable affecting PE values or shape: geometry, text
length, frequency settings, reference-image ordering and offsets, circular
settings, and video/audio positions or frame rates as applicable.
`add_shape()` records all four ggml dimensions, including trailing singleton
dimensions. `add_shapes()` records an ordered list of reference shapes.
`add(sd::Tensor<T>)` records both shape and values for explicit position inputs.
Keys compare their complete encoded fields rather than a data pointer or hash.

Use distinct names for PE inputs that coexist in a graph, such as Anima's
adapter query and key PE. `make_input()` defers reading the CPU data until graph
execution: do not replace a named entry while an unfinished graph references
it. Adding or updating other named entries does not invalidate its storage.
Return tensor shapes with the contiguous dimension first, and preserve each
model's existing F32 values, layout, and final transformations.

## Current integrations

The cache is used by MiniT2I, the Flux runner and its variants, Flux3, Qwen Image,
Qwen Image 2.1, Z-Image and L2P, Anima, Boogu, ERNIE Image, Fibo2, Ideogram4,
Iris, Krea2, Lens, LLaDA Image, MAGE Flow, Ming Image, MOSS Image, PiD, PixArt,
Wan, Hunyuan Video, LingBot Video, and LTX audio/video.

Position embeddings stored as model weights and graph-side RoPE operations
remain on their existing paths, including MiniMax H3 and SenseNova U1.
