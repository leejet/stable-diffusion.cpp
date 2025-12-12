const std::string html_content = R"xxx(
<!DOCTYPE html>
<html>

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="shortcut icon"
        href="data:image/png;base64,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" />
    <title>SDCPP Server</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100vh;
            margin: 0;
            background-color: #f0f0f0;
            color: #333;
        }

        .container {
            display: flex;
            flex-direction: column;
            align-items: center;
            width: 90%;
            max-width: 2000px;
            background: white;
            padding-inline: 20px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            gap: 1rem;
        }

        .header {
            text-align: center;
        }

        .section {
            width: 100%;
        }

        .section h2 {
            margin-bottom: 10px;
        }

        .input-group {
            display: flex;
            flex-direction: column;
            margin-bottom: 10px;
        }

        .input-group label {
            margin-bottom: 5px;
            font-weight: bold;
        }

        .prompt-input,
        .param-input {
            width: 100%;
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 5px;
            margin-bottom: 10px;
            box-sizing: border-box;
        }

        .line {
            display: flex;
            justify-content: space-between;
            width: 100%;
        }

        .line .input-group {
            width: 48%;
        }

        canvas {
            width: 100%;
            height: 100%;
        }

        .imageFrame:active {
            canvas {
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                width: auto;
                height: auto;
                min-width: 64px;
                min-height: 64px;
                max-width: 100vw;
                max-height: 100vh;
                z-index: 2;
            }

            &::before {
                content: " ";
                position: absolute;
                pointer-events: none;
                content: "";
                z-index: 1;
                display: block;
                position: fixed;
                top: 0;
                height: 100vh;
                right: 0;
                left: 0;
                background-color: rgba(0, 0, 0, .851);
                animation: fadeIn .3s;
            }
        }

        .left-section,
        .right-section {
            width: 100%;
        }

        .collapsible {
            background-color: #eee;
            color: #444;
            cursor: pointer;
            padding: 10px;
            width: 100%;
            border: none;
            text-align: left;
            outline: none;
            font-size: 15px;
            border-radius: 5px;
            margin-bottom: 10px;
        }

        .content {
            padding: 0 18px;
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.2s ease-out;
        }

        #model-id {
            display: inline-block;
            background: lightgray;
            margin-bottom: 1rem;
            padding: 5px 10px;
            border-radius: 5px;
        }

        button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            height: 3rem;
        }

        progress {
            width: 100%;
        }

        button:hover {
            background-color: #45a049;
        }

        .status {
            margin-top: 5px;
            font-size: 18px;
        }

        .container {
            min-height: 512px;
            height: 75%;
        }

        .left-section {
            overflow-y: auto;
            height: 100%;
        }

        .imageFrame {
            margin-top: auto;
            border: 1px solid #ccc;
            width: 512px;
            height: 512px;
            max-width: 90vw;
            max-height: 90vw;
        }

        .right-section {
            display: flex;
            align-items: center;
            justify-content: center;
            flex-direction: column;
        }

        @media (min-width: 768px) {
            .container {
                flex-direction: row;
            }

            .left-section,
            .right-section {
                width: 50%;
                height: 100%;
            }

            .imageFrame {
                max-width: 45vw;
                max-height: 45vw;
            }
        }

        progress {
            border: none;
            width: 100%;
            height: 8px;
            margin-inline: auto;
        }

        progress::-webkit-progress-value {
            background-color: blue;
        }

        .bars {
            margin-bottom: 10%;
            height: max-content;
            display: flex;
            width: 80%;
            flex-direction: column;
            margin-top: auto;
        }
    </style>
</head>
)xxx"
R"xxx(

<body>
    <div class="header">
        <h1>SDCPP Server</h1>
        <p>Model:<span id="model-id"></span></p>
    </div>
    <div class="container">
        <div class="left-section">
            <div class="section">
                <div class="input-group">
                    <label for="prompt">Prompt:</label>
                    <textarea id="prompt" class="prompt-input"></textarea>
                </div>
                <div class="input-group">
                    <label for="neg_prompt">Negative Prompt:</label>
                    <textarea id="neg_prompt" class="prompt-input"></textarea>
                </div>
                <button onclick="generateImage()">Generate</button>
            </div>
            <div class="section">
                <h2>Settings</h2>
                <div class="line">
                    <div class="input-group">
                        <label for="lora_model">Select LoRA:</label>
                        <select id="lora_model" class="param-input"></select>
                    </div>
                    <button onclick="addLora()">Add LoRA</button>
                </div>
                <div class="line">
                    <div class="input-group">
                        <label for="width">Width:</label>
                        <input type="number" id="width" class="param-input">
                    </div>
                    <div class="input-group">
                        <label for="height">Height:</label>
                        <input type="number" id="height" class="param-input" value=1>
                    </div>
                </div>
                <div class="line">
                    <div class="input-group">
                        <label for="seed">Seed:</label>
                        <input type="number" id="seed" class="param-input">
                    </div>
                    <div class="input-group">
                        <label for="batch_count">Batch Count:</label>
                        <input type="number" id="batch_count" class="param-input">
                    </div>
                </div>
                <div class="line">
                    <div class="input-group">
                        <label for="cfg_scale">CFG Scale:</label>
                        <input type="number" id="cfg_scale" class="param-input">
                    </div>
                    <div class="input-group">
                        <label for="guidance">Distill Guidance:</label>
                        <input type="number" id="guidance" class="param-input">
                    </div>
                </div>
                <div class="line">
                    <div class="input-group">
                        <label for="steps">Steps:</label>
                        <input type="number" id="steps" class="param-input">
                    </div>
                    <div class="input-group">
                        <label for="sample_method">Sample Method:</label>
                        <select id="sample_method" class="param-input"></select>
                    </div>
                </div>
                <div class="line">
                    <div class="input-group">
                        <label for="preview_mode">Preview:</label>
                        <select id="preview_mode" class="param-input">
                            <option value="" selected>unsupported</option>
                        </select>
                    </div>
                    <div class="input-group">
                        <label for="preview_interval">Preview interval:</label>
                        <input type="number" id="preview_interval" class="param-input">
                    </div>
                </div>
            </div>
            <button class="collapsible">Advanced</button>
            <div id="adv-params" class="content">
                <p><strong>Note:</strong> Changing these parameters may cause a longer wait time due to the models
                    reloading. Please use these parameters carefully.</p>
                <div class="input-group">
                    <label for="schedule_method">Schedule:</label>
                    <select id="schedule_method" class="param-input"></select>
                </div>
                <div class="line">
                    <div class="input-group">
                        <label for="keep_vae_on_cpu">VAE on CPU:</label>
                        <input type="checkbox" id="keep_vae_on_cpu" class="param-input">
                    </div>
                    <div class="input-group">
                        <label for="kep_clip_on_cpu">Clip on CPU:</label>
                        <input type="checkbox" id="kep_clip_on_cpu" class="param-input">
                    </div>
                </div>
                <div class="line">
                    <div class="input-group">
                        <label for="vae_tiling">VAE Tiling:</label>
                        <input type="checkbox" id="vae_tiling" class="param-input">
                    </div>
                    <div class="input-group">
                        <label for="tae_decode">Use fast TAE:</label>
                        <input type="checkbox" id="tae_decode" class="param-input">
                    </div>
                </div>
                <div id="model-loader">
                    <h3>Load new model</h3>
                    <div class="input-group">
                        <label for="model">Model:</label>
                        <select id="model" class="param-input">
                            <option value="-1" selected>keep current</option>
                            <option value="-2">None</option>
                        </select>
                    </div>
                    <div class="input-group">
                        <label for="diff-model">Diffusion Model:</label>
                        <select id="diff-model" class="param-input">
                            <option value="-1" selected>keep current</option>
                            <option value="-2">None</option>
                        </select>
                    </div>
                    <div class="line">
                        <div class="input-group">
                            <label for="clip_l">Clip_L:</label>
                            <select id="clip_l" class="param-input">
                                <option value="-1" selected>keep current</option>
                                <option value="-2">None</option>
                            </select>
                        </div>
                        <div class="input-group">
                            <label for="clip_g">Clip_G:</label>
                            <select id="clip_g" class="param-input">
                                <option value="-1" selected>keep current</option>
                                <option value="-2">None</option>
                            </select>
                        </div>
                        <div class="input-group">
                            <label for="t5xxl">T5 XXL:</label>
                            <select id="t5xxl" class="param-input">
                                <option value="-1" selected>keep current</option>
                                <option value="-2">None</option>
                            </select>
                        </div>
                    </div>
                    <div class="line">
                        <div class="input-group">
                            <label for="vae">VAE:</label>
                            <select id="vae" class="param-input">
                                <option value="-1" selected>keep current</option>
                                <option value="-2">None</option>
                            </select>
                        </div>
                        <div class="input-group">
                            <label for="tae">TAE:</label>
                            <select id="tae" class="param-input">
                                <option value="-1" selected>keep current</option>
                                <option value="-2">None</option>
                            </select>
                        </div>
                    </div>
                    <div class="input-group">
                        <label for="type">Force Quantization:</label>
                        <select id="type" class="param-input">
                            <option value="" selected>No</option>
                        </select>
                    </div>
                </div>
            </div>
        </div>
        <div class="right-section">
            <div class="imageFrame">
                <canvas id="imageCanvas" width="512" height="512"></canvas>
            </div>
            <a id="downloadLink" style="display: none;" download="generated_image.png">Download Image</a>
            <div class="bars">
                <progress id="load_progress" value="0" max="100"> 0% </progress>
                <progress id="work_progress" value="0" max="100"> 0% </progress>
                <progress id="vae_progress" value="0" max="100"> 0% </progress>
            </div>
        </div>
    </div>
    <div class="status">
        <p>Current task status: <span id="status"> -- </span> | Queue: <span id="queued_tasks">0</span></p>
    </div>
    )xxx"
    R"xxx(
    <script>
        async function addLora() {
            const lora_str = document.getElementById('lora_model').value;
            if (lora_str) {
                const prompt = document.getElementById('prompt');
                prompt.value += "<lora:" + lora_str + ":1>";
            }
        }
        let queued_tasks = 0;
        async function update_queue() {
            const display = document.getElementById('queued_tasks');
            display.innerHTML = queued_tasks;
        }
        const modelIdElement = document.getElementById('model-id');
        async function fetchModelId() {
            const response = await fetch('model');
            const data = await response.json();
            let modelIdText = '';
            if (data.model) {
                modelIdText += `${data.model}`;
                if (data.diffusion_model) {
                    modelIdText += "|";
                }
            }
            if (data.diffusion_model) {
                modelIdText += `${data.diffusion_model}`;
            }
            if (data.clip_l || data.clip_g || data.t5xxl || data.tae || data.vae) {
                modelIdText += ' ('
                if (data.clip_l) {
                    modelIdText += `clip_l: ${data.clip_l}`
                }
                if (data.clip_g) {
                    modelIdText += `, clip_g: ${data.clip_g}`
                }
                if (data.t5xxl) {
                    modelIdText += `, t5xxl: ${data.t5xxl}`
                }
                if (data.tae) {
                    modelIdText += `, tae: ${data.tae}`
                }
                if (data.vae) {
                    modelIdText += `, vae: ${data.vae}`
                }
                modelIdText += ')'
            }
            modelIdElement.textContent = modelIdText;
        }
        async function fetchSampleMethods() {
            const response = await fetch('sample_methods');
            const data = await response.json();
            const select = document.getElementById('sample_method');
            data.forEach(method => {
                const option = document.createElement('option');
                option.value = method;
                option.textContent = method;
                select.appendChild(option);
            });
        }
        async function fetchSchedules() {
            const response = await fetch('schedules');
            const data = await response.json();
            const select = document.getElementById('schedule_method');
            data.forEach(schedule => {
                const option = document.createElement('option');
                option.value = schedule;
                option.textContent = schedule;
                select.appendChild(option);
            });
        }
        async function fetchPreviewMethods() {
            const response = await fetch('previews');
            const data = await response.json();
            const select = document.getElementById('preview_mode');
            if (data) {
                select.innerHTML = '';
                data.forEach(preview => {
                    const option = document.createElement('option');
                    option.value = preview;
                    option.textContent = preview;
                    select.appendChild(option);
                });
            }
        }
        async function fetchTypes() {
            const response = await fetch('types');
            const data = await response.json();
            const select = document.getElementById('type');
            if (data) {
                data.forEach(type => {
                    const option = document.createElement('option');
                    option.value = type;
                    option.textContent = type;
                    select.appendChild(option);
                });
            }
        }
        async function fetchModels() {
            const response = await fetch('models');
            const data = await response.json();
            const modelsSelect = document.getElementById('model');
            if (data.models.length > 0) {
                data.models.forEach(model => {
                    const option = document.createElement('option');
                    option.value = model.id;
                    option.textContent = model.name;
                    modelsSelect.appendChild(option);
                });
            } else {
                modelsSelect.options.length = 1;
                const currentOption = modelsSelect.options[0];
                currentOption.select = true;
                currentOption.value = "-1";
                currentOption.textContent = "unavailable";
            }
            const diffModelsSelect = document.getElementById('diff-model');
            if (data.diffusion_models.length > 0) {
                data.diffusion_models.forEach(model => {
                    const option = document.createElement('option');
                    option.value = model.id;
                    option.textContent = model.name;
                    diffModelsSelect.appendChild(option);
                });
            } else {
                diffModelsSelect.options.length = 1;
                const currentOption = diffModelsSelect.options[0];
                currentOption.select = true;
                currentOption.value = "-1";
                currentOption.textContent = "unavailable";
            }
            const clipLSelect = document.getElementById('clip_l');
            if (data.text_encoders.length > 0) {
                data.text_encoders.forEach(encoder => {
                    const option = document.createElement('option');
                    option.value = encoder.id;
                    option.textContent = encoder.name;
                    clipLSelect.appendChild(option);
                });
            } else {
                clipLSelect.options.length = 1;
                const currentOption = clipLSelect.options[0];
                currentOption.select = true;
                currentOption.value = "-1";
                currentOption.textContent = "unavailable";
            }
            const clipGSelect = document.getElementById('clip_g');
            if (data.text_encoders.length > 0) {
                data.text_encoders.forEach(encoder => {
                    const option = document.createElement('option');
                    option.value = encoder.id;
                    option.textContent = encoder.name;
                    clipGSelect.appendChild(option);
                });
            } else {
                clipGSelect.options.length = 1;
                const currentOption = clipGSelect.options[0];
                currentOption.select = true;
                currentOption.value = "-1";
                currentOption.textContent = "unavailable";
            }
            const t5xxlSelect = document.getElementById('t5xxl');
            if (data.text_encoders.length > 0) {
                data.text_encoders.forEach(encoder => {
                    const option = document.createElement('option');
                    option.value = encoder.id;
                    option.textContent = encoder.name;
                    t5xxlSelect.appendChild(option);
                });
            } else {
                t5xxlSelect.options.length = 1;
                const currentOption = t5xxlSelect.options[0];
                currentOption.select = true;
                currentOption.value = "-1";
                currentOption.textContent = "unavailable";
            }
            const vaeSelect = document.getElementById('vae');
            if (data.vaes.length > 0) {
                data.vaes.forEach(ae => {
                    const option = document.createElement('option');
                    option.value = ae.id;
                    option.textContent = ae.name;
                    vaeSelect.appendChild(option);
                });
            } else {
                vaeSelect.options.length = 1;
                const currentOption = vaeSelect.options[0];
                currentOption.select = true;
                currentOption.value = "-1";
                currentOption.textContent = "unavailable";
            }
            const taeSelect = document.getElementById('tae');
            if (data.taes.length > 0) {
                data.taes.forEach(ae => {
                    const option = document.createElement('option');
                    option.value = ae.id;
                    option.textContent = ae.name;
                    taeSelect.appendChild(option);
                });
            } else {
                taeSelect.options.length = 1;
                const currentOption = taeSelect.options[0];
                currentOption.select = true;
                currentOption.value = "-1";
                currentOption.textContent = "unavailable";
            }
            const loraSelect = document.getElementById('lora_model');
            if (data.loras.length > 0) {
                data.loras.forEach(lora => {
                    const option = document.createElement('option');
                    option.value = lora;
                    option.textContent = lora;
                    loraSelect.appendChild(option);
                });
            } else {
                loraSelect.options.length = 1;
                const currentOption = loraSelect.options[0];
                currentOption.select = true;
                currentOption.value = null;
                currentOption.textContent = "unavailable";
            }
        }
        async function fetchParams() {
            const response = await fetch('params');
            const data = await response.json();
            document.getElementById('prompt').value = data.generation_params.prompt;
            document.getElementById('neg_prompt').value = data.generation_params.negative_prompt;
            document.getElementById('width').value = data.generation_params.width;
            document.getElementById('height').value = data.generation_params.height;
            document.getElementById('cfg_scale').value = data.generation_params.cfg_scale;
            document.getElementById('guidance').value = data.generation_params.guidance;
            document.getElementById('steps').value = data.generation_params.sample_steps;
            document.getElementById('sample_method').value = data.generation_params.sample_method;
            document.getElementById('seed').value = data.generation_params.seed;
            document.getElementById('batch_count').value = data.generation_params.batch_count;
            document.getElementById('schedule_method').value = data.context_params.schedule;
            document.getElementById('keep_vae_on_cpu').checked = data.context_params.keep_vae_on_cpu;
            document.getElementById('kep_clip_on_cpu').checked = data.context_params.kep_clip_on_cpu;
            document.getElementById('vae_tiling').checked = data.context_params.vae_tiling;
            document.getElementById('tae_decode').checked = !(data.taesd_preview);
            if (data.generation_params.preview_method) {
                document.getElementById('preview_mode').value = data.generation_params.preview_method;
            }
            if (data.generation_params.preview_interval) {
                document.getElementById('preview_interval').value = data.generation_params.preview_interval;
            }
        }
        fetchSampleMethods();
        fetchSchedules();
        fetchPreviewMethods();
        fetchModels();
        fetchModelId();
        fetchParams();
        fetchTypes();
        // )xxx" R"xxx(
        async function generateImage() {
            queued_tasks++;
            update_queue();
            const prompt = document.getElementById('prompt').value;
            const neg_prompt = document.getElementById('neg_prompt').value;
            const width = document.getElementById('width').value;
            const height = document.getElementById('height').value;
            const cfg_scale = document.getElementById('cfg_scale').value;
            const steps = document.getElementById('steps').value;
            const guidance = document.getElementById('guidance').value;
            const sample_method = document.getElementById('sample_method').value;
            const seed = document.getElementById('seed').value;
            const batch_count = document.getElementById('batch_count').value;
            const schedule_method = document.getElementById('schedule_method').value;
            const model = document.getElementById('model').value;
            const diff_model = document.getElementById('diff-model').value;
            const clip_l = document.getElementById('clip_l').value;
            const clip_g = document.getElementById('clip_g').value;
            const t5xxl = document.getElementById('t5xxl').value;
            const vae = document.getElementById('vae').value;
            const tae = document.getElementById('tae').value;
            const keep_vae_on_cpu = document.getElementById('keep_vae_on_cpu').checked;
            const kep_clip_on_cpu = document.getElementById('kep_clip_on_cpu').checked;
            const vae_tiling = document.getElementById('vae_tiling').checked;
            const tae_decode = document.getElementById('tae_decode').checked;
            const preview_mode = document.getElementById('preview_mode').value;
            const preview_interval = document.getElementById('preview_interval').value;
            const type = document.getElementById('type').value;
            const canvas = document.getElementById('imageCanvas');
            const ctx = canvas.getContext('2d');
            const downloadLink = document.getElementById('downloadLink');
            const requestBody = {
                prompt: prompt,
                negative_prompt: neg_prompt,
                ...(width && { width: parseInt(width) }),
                ...(height && { height: parseInt(height) }),
                guidance_params: {
                    ...(cfg_scale && { cfg_scale: parseFloat(cfg_scale) }),
                    ...(guidance && { guidance: parseFloat(guidance) }),
                },
                ...(steps && { sample_steps: parseInt(steps) }),
                ...(sample_method && { sample_method: sample_method }),
                ...(seed && { seed: parseInt(seed) }),
                ...(batch_count && { batch_count: parseInt(batch_count) }),
                ...(schedule_method && { schedule: schedule_method }),
                ... { model: parseInt(model) },
                ... { diffusion_model: parseInt(diff_model) },
                ... { clip_l: parseInt(clip_l) },
                ... { clip_g: parseInt(clip_g) },
                ... { t5xxl: parseInt(t5xxl) },
                ... { vae: parseInt(vae) },
                ... { tae: parseInt(tae) },
                ... { keep_vae_on_cpu: keep_vae_on_cpu },
                ... { kep_clip_on_cpu: kep_clip_on_cpu },
                ... { vae_tiling: vae_tiling },
                ... { tae_decode: tae_decode },
                ...(preview_mode && { preview_mode: preview_mode }),
                ...(preview_interval && { preview_interval: parseInt(preview_interval) }),
                ... (type && { type: type }),
            };
            const response = await fetch('txt2img', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestBody)
            });
            const data = await response.json();
            const taskId = data.task_id;
            let status = 'Pending';
            while (status !== 'Done' && status !== 'Failed') {
                const statusResponse = await fetch(`result?task_id=${taskId}`);
                const statusData = await statusResponse.json();
                if (status == 'Pending' && statusData.status != status) {
                    setTimeout(() => {
                        fetchModelId();
                        const modelsSelect = document.getElementById('model');
                        const diffModelsSelect = document.getElementById('diff-model');
                        const clipLSelect = document.getElementById('clip_l');
                        const clipGSelect = document.getElementById('clip_g');
                        const t5xxlSelect = document.getElementById('t5xxl');
                        const vaeSelect = document.getElementById('vae');
                        const taeSelect = document.getElementById('tae');
                        modelsSelect.selectedIndex = 0;
                        diffModelsSelect.selectedIndex = 0;
                        clipLSelect.selectedIndex = 0;
                        clipGSelect.selectedIndex = 0;
                        t5xxlSelect.selectedIndex = 0;
                        vaeSelect.selectedIndex = 0;
                        taeSelect.selectedIndex = 0;
                        document.getElementById("load_progress").value = 0;
                        document.getElementById("work_progress").value = 0;
                        document.getElementById("vae_progress").value = 0;
                    }, 0);
                }
                status = statusData.status;
                if (status !== "Pending" && status !== "Loading") {
                    const progressBar = document.getElementById("load_progress");
                    progressBar.value = 1;
                    progressBar.max = 1;
                    progressBar.innerHTML = "100%";
                    progressBar.style.display = 'inline-block';
                }
                const progressBar = status === "Loading" ? document.getElementById("load_progress") : status === "Working" ? document.getElementById("work_progress") : document.getElementById("vae_progress");
                document.getElementById('status').innerHTML = status;
                if (status !== 'Done' && statusData.step >= 0) {
                    progressBar.value = statusData.step;
                    progressBar.max = statusData.steps ?? steps;
                    progressBar.innerHTML = Math.floor(100 * statusData.step / statusData.steps) + "%";
                    progressBar.style.display = 'inline-block';
                }
                if (status === 'Done' || (status === 'Working' && statusData.data.length > 0)) {
                    const imageData = statusData.data[0].data;
                    const width = statusData.data[0].width;
                    const height = statusData.data[0].height;
                    const img = new Image();
                    img.src = `data:image/png;base64,${imageData}`;
                    img.onload = () => {
                        const imgRatio = img.width / img.height;
                        canvas.width = Math.max(img.width, img.height);
                        canvas.height = canvas.width;
                        let sourceX, sourceY, sourceWidth, sourceHeight;
                        if (imgRatio > 1) {
                            sourceX = 0;
                            sourceY = (img.height - img.width) / 2;
                            sourceWidth = img.width;
                            sourceHeight = img.width;
                        } else {
                            sourceX = (img.width - img.height) / 2;
                            sourceY = 0;
                            sourceWidth = img.height;
                            sourceHeight = img.height;
                        }
                        ctx.clearRect(0, 0, canvas.width, canvas.height);
                        ctx.drawImage(img, sourceX, sourceY, sourceWidth, sourceHeight, 0, 0, canvas.width, canvas.height);
                        downloadLink.href = img.src;
                        downloadLink.style.display = 'inline-block';
                    };
                } else if (status === 'Failed') {
                    alert('Image generation failed');
                }
                await new Promise(resolve => setTimeout(resolve, 100));
            }
            document.getElementById("load_progress").value = document.getElementById("load_progress").max;
            document.getElementById("work_progress").value = document.getElementById("work_progress").max;
            document.getElementById("vae_progress").value = document.getElementById("vae_progress").max;
            queued_tasks--;
            update_queue();
        }
        document.querySelectorAll('.prompt-input,.param-input').forEach(input => {
            input.addEventListener('keydown', function (event) {
                if (event.key === 'Enter') {
                    event.preventDefault();
                    generateImage();
                }
            });
        });
        var coll = document.getElementsByClassName("collapsible");
        for (let i = 0; i < coll.length; i++) {
            coll[i].addEventListener("click", function () {
                this.classList.toggle("active");
                var content = this.nextElementSibling;
                content.style.transition = "max-height 0.3s ease-out"; // Add a transition effect
                if (content.style.maxHeight) {
                    content.style.maxHeight = null;
                } else {
                    content.style.maxHeight = content.scrollHeight + "px";
                    // Scroll to the top of the content
                    content.scrollTop = 0;
                }
            });
        }
    </script>
</body>

</html>
)xxx";