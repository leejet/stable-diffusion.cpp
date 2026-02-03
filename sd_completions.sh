_sd_completions() {
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"

    opts="-diffusion -cli -o --output -style --preview-path --preview-interval --output-begin-idx -negative --canny --convert-name -v --verbose --color --taesd-preview-only --preview --preview-noisy --mode --preview -h --help -m --model --clip -l --clip -g --clip -vision --t --llm -image -small --llm --qwen --llm --qwen --llm --diffusion-model --high-noise-diffusion-model --vae --taesd --tae --taesd --control-net --embd-dir --lora-model-dir --tensor-type-rules --photo-maker --upscale-model -t --threads --chroma-t -mask-pad --vae-tile-overlap --flow-shift --vae-tiling --force-sdxl-vae-conv-scale --offload-to-cpu --mmap -map --control-net-cpu --clip-on-cpu --vae-on-cpu --diffusion-fa --diffusion-conv-direct --vae-conv-direct --circular --circularx -axis --circulary -axis --chroma-disable-dit-mask --qwen-image-zero-cond-t --chroma-enable-t -mask --type --rng -webui --sampler-rng --rng --prediction --lora-apply-mode --vae-tile-size --vae-relative-tile-size --vae-tile-size -p --prompt -n --negative-prompt -i --init-img --end-img --mask --control-image --control-video --pm-id-images-dir --pm-id-embed-path --height --width --steps --high-noise-steps --clip-skip -b --batch-count --video-frames --fps --timestep-shift --upscale-repeats --upscale-tile-size --cfg-scale --img-cfg-scale -pix --cfg-scale --guidance --slg-scale --skip-layer-start --skip-layer-end --eta --high-noise-cfg-scale --high-noise-img-cfg-scale -pix --cfg-scale --high-noise-guidance --high-noise-slg-scale --high-noise-skip-layer-start --high-noise-skip-layer-end --high-noise-eta --strength --pm-style-strength --control-strength --moe-boundary --high-noise-steps --vace-strength --increase-ref-index --disable-auto-resize-ref-image -s --seed --sampling-method --high-noise-sampling-method --scheduler --sigmas -separated --skip-layers --high-noise-skip-layers -r --ref-image --cache-mode -dit -level --cache-option -separated -dit --cache-preset -dit --scm-mask -dit -separated --scm-policy"

    case "$prev" in
        --diffusion-model)
            COMPREPLY=( $(compgen -f -X '!*.gguf' -- "$cur") $(compgen -d -- "$cur") )
            return 0
            ;;
        -m)
            COMPREPLY=( $(compgen -f -X '!*.safetensors' -- "$cur") $(compgen -d -- "$cur") )
            return 0
            ;;
        *)
            COMPREPLY=( $(compgen -W "${opts}" -- "$cur") )
            return 0
            ;;
    esac
}
_sd_server_comps() {
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"

    opts="-diffusion -server -l --listen-ip --serve-html-path --listen-port -v --verbose --color -h --help -m --model --clip -l --clip -g --clip -vision --t --llm -image -small --llm --qwen --llm --qwen --llm --diffusion-model --high-noise-diffusion-model --vae --taesd --tae --taesd --control-net --embd-dir --lora-model-dir --tensor-type-rules --photo-maker --upscale-model -t --threads --chroma-t -mask-pad --vae-tile-overlap --flow-shift --vae-tiling --force-sdxl-vae-conv-scale --offload-to-cpu --mmap -map --control-net-cpu --clip-on-cpu --vae-on-cpu --diffusion-fa --diffusion-conv-direct --vae-conv-direct --circular --circularx -axis --circulary -axis --chroma-disable-dit-mask --qwen-image-zero-cond-t --chroma-enable-t -mask --type --rng -webui --sampler-rng --rng --prediction --lora-apply-mode --vae-tile-size --vae-relative-tile-size --vae-tile-size -p --prompt -n --negative-prompt -i --init-img --end-img --mask --control-image --control-video --pm-id-images-dir --pm-id-embed-path --height --width --steps --high-noise-steps --clip-skip -b --batch-count --video-frames --fps --timestep-shift --upscale-repeats --upscale-tile-size --cfg-scale --img-cfg-scale -pix --cfg-scale --guidance --slg-scale --skip-layer-start --skip-layer-end --eta --high-noise-cfg-scale --high-noise-img-cfg-scale -pix --cfg-scale --high-noise-guidance --high-noise-slg-scale --high-noise-skip-layer-start --high-noise-skip-layer-end --high-noise-eta --strength --pm-style-strength --control-strength --moe-boundary --high-noise-steps --vace-strength --increase-ref-index --disable-auto-resize-ref-image -s --seed --sampling-method --high-noise-sampling-method --scheduler --sigmas -separated --skip-layers --high-noise-skip-layers -r --ref-image --cache-mode -dit -level --cache-option -separated -dit --cache-preset -dit --scm-mask -dit -separated --scm-policy"

    case "$prev" in
        --diffusion-model)
            COMPREPLY=( $(compgen -f -X '!*.gguf' -- "$cur") $(compgen -d -- "$cur") )
            return 0
            ;;
        -m)
            COMPREPLY=( $(compgen -f -X '!*.safetensors' -- "$cur") $(compgen -d -- "$cur") )
            return 0
            ;;
        *)
            COMPREPLY=( $(compgen -W "${opts}" -- "$cur") )
            return 0
            ;;
    esac
}

complete -F _sd_completions sd-cli
complete -F _sd_server_comps sd-server
