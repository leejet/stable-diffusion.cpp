import { html, render, useSignal, signal, useEffect } from './preact.js';

const modes = ["Text2Image", "Image2Image"]

const runtime = signal({
    generating: false,
    is_canceling: false,
    new_generation: false,
    models: ["Default Model"],
    current_image: 0,
    images: [],
    placeholder_image: '', // base64 blank image for placeholder
    show_settings: false,
    show_preview: false,
    init_image: null
});

const params = signal({
    model: 0,
    prompt: "",
    negative_prompt: "",
    cfg_scale: 7.0,
    width: 512,
    height: 512,
    steps: 20,
    seed: 42,
    random_seed: false,
    sampler: 0,
    schedule: 0,
    batch_count: 1,
    upscale: false,
    denoise_scale: 0.4,
    mode: 0,
    vae_tiling: false,
    clip_skip: -1,
    dark_mode: false
});

const model_state = signal({
    loaded: false,
    decoding: false,
    state_info: '',
    batch_index: -1,
    batch_count: -1
});

const preview_state = signal({
    scale: 1,
    panX: 0,
    panY: 0,
    isPanning: false,
    startX: 0,
    startY: 0
});

// fetch saved params
let params_str = localStorage.getItem('params__sdcpp');
if(params_str) {
    let params__ = JSON.parse(params_str);
    params.value = { ...params.value, ...params__ };
}

var controller;

function App () {
    useEffect(async () => {
        // update placeholder for images
        const response = await fetch("/placeholder", {
            method: 'POST',
            body: JSON.stringify({ ...params.value, seed: -1 }),
            headers: { 'Connection': 'keep-alive',
                'Content-Type': 'application/json' }
        });
        runtime.value = {...runtime.value, placeholder_image: 'data:image/png;base64,' + (await response.json()).data }
    }, [params.value.width, params.value.height, params.value.dark_mode]);

    // autosave params
    useEffect(() => {
        localStorage.setItem('params__sdcpp', JSON.stringify(params.value));
    }, [params.value]);

    useEffect(() => {
        if(params.value.dark_mode) {
            document.querySelector("body").classList.add("dark-mode");
        } else {
            document.querySelector("body").classList.remove("dark-mode");
        }
    }, [params.value.dark_mode]);

    // ui elements
    const TextAreaField = ({name, placeholder, param, oninput}) => {
        const field_st = useSignal({ classes: "" });
        const updateFocus = () => field_st.value = { classes: "focus" };
        const updateBlur = () => {
            if(param === "") {
                field_st.value = { classes: "" };
            }
        };
        return html`
        <div class="text-area-field-box">
            <textarea class="${param === "" ? field_st.value.classes : 'focus'}" name="${name}" value="${param}" onInput=${oninput} onFocus=${updateFocus} onBlur=${updateBlur}/>
            <span data-placeholder="${placeholder}"></span>
        </div>`;
    }

    const RangeField = ({name, placeholder, min, max, step, param, oninput}) => {
        return html`
        <div class="range-slider">
            <span style="display: block;">${placeholder}</span>
            <input className="range-slider__range" type="range" min="${min}" max="${max}" name="${name}" step="${step}" value="${param}" onInput=${oninput} />
            <span className="range-slider__value">${param}</span>
        </div>`;
    }

    const TextField = ({name, placeholder, param, oninput, type}) => {
        const field_st = useSignal({ classes: "" });
        const updateFocus = () => {
            field_st.value = { classes: "focus" };
        };
        const updateBlur = () => {
            if(param === "") {
                field_st.value = { classes: "" };
            }
        };
        return html`
        <div class="input-text-field-box">
            <input type="text" class="${param === "" ? field_st.value.classes : 'focus'}"  type="${type??"text"}" name="${name}" value="${param}" onInput=${oninput} onFocus=${updateFocus} onBlur=${updateBlur}/>
            <span data-placeholder="${placeholder}"></span>
        </div>`;
    }

    const CheckBoxField = ({name, placeholder, param, oninput}) => {
        return html`
        <div class="check-field">
            <input class="checkbox-input" type="checkbox" name="${name}" value="0" checked=${param} onInput=${oninput} />
            <span>${placeholder}</span>
        </div>`;
    }

    const SelectField = ({name, placeholder, param, options}) => {
        const onSelect = (el) => {
            params.value = { ...params.value, [name]: Math.floor(parseFloat(el.target.id)) };
        };
        return html`
        <div class="select-box">
            <span>${placeholder}</span>
            <div class="select-current" tabindex="1">
                ${
                    options.map((option, idx) => html`
                        <div class="select-value">
                            <input class="select-input" type="radio" value="${idx}" name="${option}" checked=${param == idx}/>
                            <p class="select-input-text">${option}</p>
                        </div>
                    `)
                }
                <svg class="select-arrow" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" x="0px" y="0px" viewBox="0 0 256 256" enable-background="new 0 0 256 256" xml:space="preserve">
                    <g><g><path fill="#fff" d="M128,194.3L10,76.8l15.5-15.1L128,164.2L230.5,61.7L246,76.8L128,194.3z"/></g></g>
                </svg>
            </div>
            <ul class="select-list disable-scrollbars">
                ${
                    options.map((option, idx) => html`
                        <li>
                            <label class="select-option" id="${idx}" onClick=${onSelect} aria-hidden="aria-hidden">${option}</label>
                        </li>
                    `)
                }
            </ul>
        </div>`;
    }

    const ImageDropField = ({name, placeholder, param}) => {
        const selectFile = (file) => {
            const reader = new FileReader();
            reader.onload = function (e) {
                const img_ = new Image();
                img_.src = e.target.result;
                runtime.value = { ...runtime.value, [name]: img_ };
            };
            reader.readAsDataURL(file);
        }
        const dropFile = (ev) => {
            ev.preventDefault();
            let files = ev.dataTransfer.files;
            if(files.length > 0) {
                selectFile(files[0]);
            }
        }
        const dragOver = (ev) => { ev.preventDefault(); }
        return html`
        <div class="${!param ? 'image-drag-drop' :'image-drag-drop-showing'}">
            <span>${placeholder}</span>
            <input type="file" id="fileInput" style="display: none;" onChange=${(el) => { if(el.target.files.length > 0){ selectFile(el.target.files[0]) } }}/>
            <div class="image-prev-dd" ondrop=${dropFile} ondragover=${dragOver} onclick=${() => {document.getElementById("fileInput").click()}}>
                ${
                    param ? html`
                    <img src=${param.src} />
                    ` : html`
                    <span>No image loaded</span>
                    `
                }
            </div>
        </div>`;
    }

    const downloadImage = (el) => {
        let link = document.createElement("a");
        link.href = runtime.value.images[runtime.value.current_image].data;
        link.download = "image_"+ runtime.value.images[runtime.value.current_image].seed + ".png";
        link.click();
    };

    const ImageGallery = () => {
        const selectImage = (el) => {
            preview_state.value = {
                scale: 1, panX: 0, panY: 0,
                isPanning: false, startX: 0, startY: 0
            };
            runtime.value = { ...runtime.value, current_image: parseInt(el.target.name), show_preview: true };
        };
        return html`
            <div class="image-gallery">
                <div class="images disable-scrollbars">
                    ${
                        runtime.value.images.map((image, idx) => html`
                            <div class="item ${image.status == 'rd' ? '' : (image.status == 'nv' ? '' : 'processing-this ') + 'not-available'}">
                                <img src="${image.data}" name="${idx}" onClick=${selectImage}/>
                            </div>
                        `)
                    }
                </div>
            </div>
        `;
    }

    const updateParams = (el) => params.value = { ...params.value, [el.target.name]: el.target.value };
    const updateParamsBool = (el) => params.value = { ...params.value, [el.target.name]: el.target.checked }
    const updateParamsInt = (el) => {
        if(el.target.value.length === 0) {
            return;
        }
        if(el.target.value.length == 1 && el.target.value === "-") {
            return;
        }
        params.value = { ...params.value, [el.target.name]: parseInt(el.target.value) }
    };
    const updateParamsFloat = (el) => {
        if(el.target.value.length === 0 || el.target.value.length == 1 && el.target.value === "-") {
            return;
        }
        params.value = { ...params.value, [el.target.name]: parseFloat(el.target.value) }
    };

    const zoomPreview = (e) => {
        e.preventDefault();
        const scaleAmount = 0.1;
        const maxScale = 3;
        const minScale = 1;
        let scale = 0;
        if(e.deltaY < 0) {
            scale = Math.min(preview_state.value.scale + scaleAmount, maxScale);
        } else {
            scale = Math.max(preview_state.value.scale - scaleAmount, minScale);
        }
        preview_state.value = {
            ...preview_state.value, scale};
    };
    const previewMouseDown = (e) => {
        preview_state.value = {
            ...preview_state.value,
            isPanning: true,
            startX: e.clientX - preview_state.value.panX,
            startY: e.clientY - preview_state.value.panY };
    };
    const previewMouseUp = (e) => {
        preview_state.value = {
            ...preview_state.value, isPanning: false};
    };
    const previewMouseMove = (e) => {
        if(preview_state.value.isPanning) {
            preview_state.value = {
                ...preview_state.value,
                panX: e.clientX - preview_state.value.startX,
                panY: e.clientY - preview_state.value.startY };
        }
    };

    // main function
    const requestGeneration = async () => {
        if(runtime.value.generating) {
            runtime.value = { ...runtime.value, generating: false, images: [] };
            controller.abort();
            await fetch("cancel");
            controller = null;
        }
        // generate placeholders
        let images = [];
        for(let i = 0; i < params.value.batch_count; i++) {
            images.push({ status: 'nv', data: runtime.value.placeholder_image });
        }

        // change state
        runtime.value = { ...runtime.value, generating: true, new_generation: true, images };
        model_state.value = {...model_state.value, loaded: false, decoding: false };

        // request generation
        controller = new AbortController();
        try {
            const response = await fetch("/txt2img", {
                method: 'POST',
                body: JSON.stringify({ ...params.value, stream: true, seed: params.value.random_seed ? -1 : params.value.seed }),
                headers: {
                    'Connection': 'keep-alive',
                    'Content-Type': 'application/json',
                    'Accept': 'text/event-stream'
                },
                signal: controller.signal
            });
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let leftover = ""; // Buffer for partially read lines
            let cont = true;
            while (cont) {
                const result = await reader.read();
                if (result.done) {
                    break;
                }
                const text = leftover + decoder.decode(result.value);
                const endsWithLineBreak = text.endsWith('\n');
                let lines = text.split('\n');
                leftover = !endsWithLineBreak ? lines.pop() : "";
                const regex = /^(\S+):\s(.*)$/gm;
                for (const line of lines) {
                    const match = regex.exec(line);
                    if (match) {
                        result[match[1]] = match[2];
                        // read full data
                        if (result.data) {
                            let chunk_data = JSON.parse(result.data);
                            if(chunk_data.type == "status") {
                                if(chunk_data.loaded) {
                                    model_state.value = {...model_state.value, loaded: true, state_info: 'Sampling ...' };
                                } else if(!chunk_data.decoding) {
                                    let progress = (chunk_data.progress_current/chunk_data.progress_total) * 100.0;
                                    document.getElementById("pb-global").style.setProperty('--porcent-prog', progress+'%');
                                    model_state.value = { ...model_state.value, state_info: 'Sampling ' + model_state.value.batch_index + "/" + model_state.value.batch_count + " - " + progress.toFixed(2)+"%" };
                                } else {
                                    model_state.value = {...model_state.value, decoding: true };
                                }
                            } else if(chunk_data.type == "image") {
                                if(params.value.random_seed) {
                                    params.value = { ...params.value, seed: chunk_data.seed };
                                }
                                let images = [];
                                for(let i = 0; i < params.value.batch_count; i++) {
                                    if(runtime.value.images[i].status == 'rd' && i != chunk_data.index) {
                                        images.push(runtime.value.images[i]);
                                        continue;
                                    }
                                    if(i == chunk_data.index) {
                                        images.push({ status: 'rd', seed: chunk_data.seed, data: 'data:image/png;base64,' + chunk_data.data  });
                                    } else {
                                        images.push({ status: 'nv' });
                                    }
                                }
                                runtime.value = { ...runtime.value, images };
                            }  else if(chunk_data.type == "sampled_image") {
                                let images = [];
                                for(let i = 0; i < params.value.batch_count; i++) {
                                    if(i == (model_state.value.batch_index - 1)) {
                                        images.push({ status: 'pg', seed: -1, data: 'data:image/png;base64,' + chunk_data.data  });
                                    } else {
                                        images.push({ status: 'nv', seed: -1, data: runtime.value.images[i].data });
                                    }
                                }
                                runtime.value = { ...runtime.value, images };
                            } else if(chunk_data.type == "new_image") {
                                model_state.value = {...model_state.value, batch_count: chunk_data.count, batch_index: chunk_data.index+1 };
                                let images = [];
                                for(let i = 0; i < params.value.batch_count; i++) {
                                    images.push({ status: i == chunk_data.index ? 'pg' : 'nv', data: runtime.value.images[i].data });
                                }
                                if(model_state.value.decoding) {
                                    for(let i = chunk_data.index; i >= 0; i--) {
                                        images[i].status = 'rd';
                                    }
                                    model_state.value = { ...model_state.value, state_info: 'Decoding ' + model_state.value.batch_index + "/" + model_state.value.batch_count };
                                    document.getElementById("pb-global").style.setProperty('--porcent-prog','0%');
                                }
                                runtime.value = { ...runtime.value, images };
                            }
                            if(chunk_data.stop) {
                                cont = false;
                                break;
                            }
                        }
                    }
                }
            }
        } catch (e) {
            if (e.name !== 'AbortError') {
                console.error("sd error: ", e);
                runtime.value = { ...runtime.value, generating: false, images: [] };
            }
        }
        runtime.value = { ...runtime.value, generating: false };
    };

    return html`
        <div>
            <div class="container">
                <h1 class="app-title">Stable Diffusion</h1>
                <div class="row-main">
                    <div class="col params disable-scrollbars">
                        <div class="row">
                            <fieldset class="col">
                                ${SelectField({name: "mode", placeholder: "Mode", param: params.value.mode, oninput: updateParams, options: modes})}
                            </fieldset>
                            <fieldset class="col">
                                ${SelectField({name: "model", placeholder: "Model", param: params.value.model, oninput: updateParams, options: runtime.value.models})}
                            </fieldset>
                        </div>
                        <fieldset style="border: none; padding:0;">
                            ${params.value.mode == 1 ? ImageDropField({name: "init_image", placeholder: "Initial image", param: runtime.value.init_image}) : ''}
                            ${TextAreaField({name: "prompt",placeholder: "Prompt", param: params.value.prompt, oninput: updateParams})}
                            ${TextAreaField({name: "negative_prompt",placeholder: "Negative Prompt", param: params.value.negative_prompt, oninput:  updateParams})}
                        </fieldset>
                        <div class="row">
                            <fieldset class="col" style="border: none; padding:0;">
                                ${SelectField({name: "sampler", placeholder: "Sampler Method", param: params.value.sampler, oninput: updateParamsInt, options: ["Euler A", "Euler", "Heun", "DPM2", "DPM++ 2S A", "DPM++ 2M", "DPM++ 2M v2", "LCM"]})}
                                ${RangeField({name: "steps", placeholder: "Steps", min: 1, max: 50, steps: 1, param: params.value.steps, oninput: updateParamsInt})}
                                ${RangeField({name: "cfg_scale", placeholder: "CFG Scale", min: 1, max: 20, step: 0.5, param: params.value.cfg_scale, oninput: updateParamsFloat})}
                                ${TextField({name: "seed", placeholder: "Seed", param: params.value.seed, oninput: updateParamsInt, type: "number"})}
                                ${CheckBoxField({name: "random_seed", placeholder: "Random", param: params.value.random_seed, oninput: updateParamsBool})}
                            </fieldset>
                            <fieldset class="col" style="border: none; padding: 0 0.6rem;">
                                ${SelectField({name: "schedule", placeholder: "Schedule", param: params.value.schedule, oninput: updateParamsInt, options: ["Default", "Discrete", "Karras", "Align your steps"]})}
                                <div class="row">
                                    <fieldset class="col">
                                        ${TextField({name: "width", placeholder: "Width", param: params.value.width, oninput: updateParamsInt, type: "number"})}
                                    </fieldset>
                                    <fieldset class="col">
                                        ${TextField({name: "height", placeholder: "Height", param: params.value.height, oninput: updateParamsInt, type: "number"})}
                                    </fieldset>
                                </div>
                                ${CheckBoxField({name: "upscale", placeholder: "Upscale", param: params.value.upscale, oninput: updateParamsBool})}
                                ${RangeField({name: "batch_count", placeholder: "Batch Count", min: 1, max: 40, step: 1, param: params.value.batch_count, oninput: updateParamsInt})}
                                ${params.value.mode == 1 ? RangeField({name: "denoise_scale", placeholder: "Denoise Scale", min: 0.0, max: 1.0, step: 0.1, param: params.value.denoise_scale, oninput: updateParamsFloat}) : ''}
                            </fieldset>
                        </div>
                    </div>
                    <fieldset class="col">
                        <div class="row-gen">
                            <button class="button-holo" style="margin-top: 10px;" role="button" onClick=${() => { runtime.value = { ...runtime.value, show_settings: true }; }}>Settings</button>
                            <button class="button-holo" style="margin-top: 10px; margin-left: 10px;" role="button" onClick=${requestGeneration}>${runtime.value.generating ? "Cancel" : "Generate"}</button>
                            ${
                                runtime.value.generating ? html`
                                <div>
                                    <div class="progress-bar ${(!model_state.value.loaded || model_state.value.decoding) ? 'indeterminate' : ''}" id="pb-global">
                                    </div>
                                    <p class="progress-text">${
                                        model_state.value.loaded ? model_state.value.state_info : 'Loading model'}</p>
                                </div>`: ''
                            }
                        </div>
                        ${ImageGallery()}
                    </fieldset>
                </div>
            </div>
            <div class="modal ${runtime.value.show_settings ? 'is-open' : ''}">
                <div class="modal-container">
                    <div class="modal-left">
                        <h1 class="modal-title">Settings</h1>
                        <fieldset class="col">
                            ${CheckBoxField({name: "dark_mode", placeholder: "Dark Mode", param: params.value.dark_mode, oninput: updateParamsBool})}

                            ${TextField({name: "clip_skip", placeholder: "Skip clip layer", param: params.value.clip_skip, oninput: updateParamsInt, type: "number"})}
                            ${CheckBoxField({name: "vae_tiling", placeholder: "VAE Tiling", param: params.value.vae_tiling, oninput: updateParamsBool})}
                        </fieldset>
                    </div>
                    <button class="icon-button" onClick=${()=>{runtime.value = { ...runtime.value, show_settings: false };}}>
                        <svg viewBox="0 0 50 50" version="1.1" id="svg4" opacity="1" sodipodi:docname="close.svg"
                            inkscape:version="1.2.2 (732a01da63, 2022-12-09)"
                            xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
                            xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
                            xmlns="http://www.w3.org/2000/svg" xmlns:svg="http://www.w3.org/2000/svg">
                            <defs id="defs8" />
                            <path
                                d="M 25 3 C 12.86158 3 3 12.86158 3 25 C 3 37.13842 12.86158 47 25 47 C 37.13842 47 47 37.13842 47 25 C 47 12.86158 37.13842 3 25 3 z M 25 5 C 36.05754 5 45 13.94246 45 25 C 45 36.05754 36.05754 45 25 45 C 13.94246 45 5 36.05754 5 25 C 5 13.94246 13.94246 5 25 5 z M 16.990234 15.990234 A 1.0001 1.0001 0 0 0 16.292969 17.707031 L 23.585938 25 L 16.292969 32.292969 A 1.0001 1.0001 0 1 0 17.707031 33.707031 L 25 26.414062 L 32.292969 33.707031 A 1.0001 1.0001 0 1 0 33.707031 32.292969 L 26.414062 25 L 33.707031 17.707031 A 1.0001 1.0001 0 0 0 32.980469 15.990234 A 1.0001 1.0001 0 0 0 32.292969 16.292969 L 25 23.585938 L 17.707031 16.292969 A 1.0001 1.0001 0 0 0 16.990234 15.990234 z"
                                id="path2" />
                        </svg>
                    </button>
                </div>
            </div>
            <div class="modal ${runtime.value.show_preview ? 'is-open' : ''}">
                <div class="modal-container-preview">
                ${runtime.value.show_preview ? html`
                    <div class="modal-left-preview">
                        <h1 class="modal-title">Image ${runtime.value.images[runtime.value.current_image].seed}</h1>
                        <button class="button-holo" style="margin-top: 10px;" role="button" onClick=${downloadImage}>
                            <svg width="40px" height="40px" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                                <path d="M3 14.25C3.41421 14.25 3.75 14.5858 3.75 15C3.75 16.4354 3.75159 17.4365 3.85315 18.1919C3.9518 18.9257 4.13225 19.3142 4.40901 19.591C4.68577 19.8678 5.07435 20.0482 5.80812 20.1469C6.56347 20.2484 7.56459 20.25 9 20.25H15C16.4354 20.25 17.4365 20.2484 18.1919 20.1469C18.9257 20.0482 19.3142 19.8678 19.591 19.591C19.8678 19.3142 20.0482 18.9257 20.1469 18.1919C20.2484 17.4365 20.25 16.4354 20.25 15C20.25 14.5858 20.5858 14.25 21 14.25C21.4142 14.25 21.75 14.5858 21.75 15V15.0549C21.75 16.4225 21.75 17.5248 21.6335 18.3918C21.5125 19.2919 21.2536 20.0497 20.6517 20.6516C20.0497 21.2536 19.2919 21.5125 18.3918 21.6335C17.5248 21.75 16.4225 21.75 15.0549 21.75H8.94513C7.57754 21.75 6.47522 21.75 5.60825 21.6335C4.70814 21.5125 3.95027 21.2536 3.34835 20.6517C2.74643 20.0497 2.48754 19.2919 2.36652 18.3918C2.24996 17.5248 2.24998 16.4225 2.25 15.0549C2.25 15.0366 2.25 15.0183 2.25 15C2.25 14.5858 2.58579 14.25 3 14.25Z"/>
                                <path d="M12 16.75C12.2106 16.75 12.4114 16.6615 12.5535 16.5061L16.5535 12.1311C16.833 11.8254 16.8118 11.351 16.5061 11.0715C16.2004 10.792 15.726 10.8132 15.4465 11.1189L12.75 14.0682V3C12.75 2.58579 12.4142 2.25 12 2.25C11.5858 2.25 11.25 2.58579 11.25 3V14.0682L8.55353 11.1189C8.27403 10.8132 7.79963 10.792 7.49393 11.0715C7.18823 11.351 7.16698 11.8254 7.44648 12.1311L11.4465 16.5061C11.5886 16.6615 11.7894 16.75 12 16.75Z"/>
                        </svg> Download</button>
                    </div>
                    <div class="modal-right-preview"
                        onWheel=${zoomPreview}
                        onMouseDown=${previewMouseDown}
                        onMouseUp=${previewMouseUp}
                        onMouseLeave=${previewMouseUp}
                        onMouseMove=${previewMouseMove}
                        style="cursor: ${preview_state.value.isPanning ? 'grabbing' : 'grab'}">
                        <img style="transform: translate(${preview_state.value.panX}px, ${preview_state.value.panY}px) scale(${preview_state.value.scale});" src="${runtime.value.images[runtime.value.current_image].data}"/>
                    </div>
                    <button class="icon-button" onClick=${()=>{runtime.value = { ...runtime.value, show_preview: false };}}>
                        <svg viewBox="0 0 50 50" version="1.1" id="svg4" opacity="1" sodipodi:docname="close.svg"
                            inkscape:version="1.2.2 (732a01da63, 2022-12-09)"
                            xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
                            xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
                            xmlns="http://www.w3.org/2000/svg" xmlns:svg="http://www.w3.org/2000/svg">
                            <defs id="defs8" />
                            <path
                                d="M 25 3 C 12.86158 3 3 12.86158 3 25 C 3 37.13842 12.86158 47 25 47 C 37.13842 47 47 37.13842 47 25 C 47 12.86158 37.13842 3 25 3 z M 25 5 C 36.05754 5 45 13.94246 45 25 C 45 36.05754 36.05754 45 25 45 C 13.94246 45 5 36.05754 5 25 C 5 13.94246 13.94246 5 25 5 z M 16.990234 15.990234 A 1.0001 1.0001 0 0 0 16.292969 17.707031 L 23.585938 25 L 16.292969 32.292969 A 1.0001 1.0001 0 1 0 17.707031 33.707031 L 25 26.414062 L 32.292969 33.707031 A 1.0001 1.0001 0 1 0 33.707031 32.292969 L 26.414062 25 L 33.707031 17.707031 A 1.0001 1.0001 0 0 0 32.980469 15.990234 A 1.0001 1.0001 0 0 0 32.292969 16.292969 L 25 23.585938 L 17.707031 16.292969 A 1.0001 1.0001 0 0 0 16.990234 15.990234 z"
                                id="path2" />
                        </svg>
                    </button>
                ` : ''}
                </div>
            </div>
        </div>`;
}

render(html`<${App}/>`, document.getElementById("app-viewport"));