# Generation results and metadata

`sd_generate_image()`, `sd_generate_video()` and `sd_upscale()` all execute dgl.
`sd_result_get_generation_info(result, index)` returns the exact UTF-8 program
submitted to the DGL compiler. It is not a JSON parameter snapshot.

Debug logging also prints the actual compiled program immediately before
execution. Use `--log-level debug` in the CLI. Each image in a batch
logs its own resolved program and seed. Source logging is independent of file
metadata embedding; tensor bindings remain separate inputs and are not dumped.

## Results and rerunning

Each generated image records its actual seed and executed source. Image batches
execute individual programs in the same context, reusing loaded models. Multiple
layers decoded from one latent share the same source. A video records one
program for the entire sequence, returned for every valid frame index.
Partial batch cancellation returns records only for completed images; full
cancellation returns no result.

Rerun a self-contained program through `sd-cli -M dgl --script FILE` or the
DGL C API. Programs with `fn main(input_image, ...)` require the original host
tensor bindings through the DGL C API. Generated programs retain component paths,
resolved options and ordinary LoRA application order. They do not embed model
weights or input media. Keep the same files, inputs and backends when comparing
numerical results.

Standalone image upscaling records the complete DGL operation, including every
requested repeat. No source diffusion settings are invented for an imported
image. There is no postprocessing JSON merge or separate replay mode.

## WebUI infotext

CLI and server derive WebUI infotext from the DGL source on a best-effort basis.
Known image flows can expose prompts, seed, dimensions, sampler, scheduler,
steps, CFG, strength, components and ordinary LoRAs. Ambiguous multi-stage fields
are omitted. Conversion never changes the executed source or causes inference
to fail. Standalone upscaling can have empty infotext.

`sd_get_generation_metadata(source)` returns UTF-8 JSON containing these
parameters and an `infotext` field in one call. It does not execute the program
or read model files. Unrecognized image flows return `{}`. The returned string
is borrowed until the next call on the same thread; copy or parse it before then.
A null result indicates an API error available through `sd_get_last_error()`.

Native server `generation_info` values are DGL strings. WebUI-compatible routes
also fill `infotexts` separately. Clients must not parse the DGL string as JSON.

## File metadata

The CLI and server image writers store native DGL and WebUI text separately:

| Container | DGL source | WebUI text |
| --- | --- | --- |
| PNG | UTF-8 `iTXt`, keyword `dgl` | UTF-8 `iTXt`, keyword `parameters` |
| JPEG | COM segment labeled `dgl` | COM segment labeled `parameters` |
| WebP / animated WebP | XMP `sdcpp:dgl` | XMP `sdcpp:parameters` |
| AVI | RIFF INFO `ICMT` | Not stored separately |
| WebM / Matroska | `COMMENT` tag | Not stored separately |

Video comments contain the DGL program. The XMP namespace is
`https://github.com/leejet/stable-diffusion.cpp/ns/1.0/`.
JPEG labels and values are separated by NUL bytes. A JPEG COM payload is limited
to 65,533 bytes including labels and terminators; an oversized record makes
encoding fail rather than silently truncating the program. Other container
capacity checks follow the same rule.

Memory encoding and file saving use the same metadata writer. Disabling metadata
still disables embedding. Previously generated files with combined WebUI text
and `SDCPP:` JSON remain readable.

DGL `image::save` (PNG) and `video::save` (AVI) also embed the exact program
source by default, using `dgl` and `ICMT` respectively. `image::encode` and
`video::encode` embed it during encoding; `media::save` writes those bytes
unchanged. WebM uses the `COMMENT` tag. These script functions store the program
itself without deriving WebUI infotext. Set the DGL context
option `embed_metadata` to `off` to omit it; `sd-cli -M dgl` forwards
`--disable-image-metadata` to that option. The switch also disables embedding
in the ordinary CLI image and video output paths. It does not affect tensor
data or the source returned by `sd_result_get_generation_info`.

Inspect metadata with `sd-cli -M metadata --image FILE`, optionally adding
`--metadata-format json` for a structured report. This command reads container
metadata without loading a model.

The C API provides the same reader through `sd_read_metadata(path, flags)`.
Use zero for the default text report, or combine `SD_METADATA_READ_JSON`,
`SD_METADATA_READ_RAW`, `SD_METADATA_READ_BRIEF` and
`SD_METADATA_READ_STRUCTURAL`. A null return indicates an error; inspect
`sd_get_last_error()`. The returned UTF-8 report is borrowed until the next
call to this function on the same thread.

For applications that encode their own files, `sd_get_metadata_chunk()` builds
PNG iTXt chunks, JPEG COM segments, XMP text or an AVI INFO chunk. It returns
borrowed bytes and their exact size; copy them before the next call on the same
thread. PNG chunks are inserted after IHDR, JPEG segments after SOI, XMP is
stored in the container's XMP field, and AVI INFO is a RIFF chunk. Empty source
returns an empty chunk. This function does not write a file.
