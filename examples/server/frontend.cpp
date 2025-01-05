const std::string html_content = R"xxx(
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
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
            margin-top: 10px;
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
            </div>
            <button onclick="generateImage()">Generate</button>
            <div class="section">
                <h2>Settings</h2>
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
                        <label for="vae_on_cpu">VAE on CPU:</label>
                        <input type="checkbox" id="vae_on_cpu" class="param-input">
                    </div>
                    <div class="input-group">
                        <label for="clip_on_cpu">Clip on CPU:</label>
                        <input type="checkbox" id="clip_on_cpu" class="param-input">
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
                            <option value="" selected>keep current</option>
                            <option value="none">None</option>
                        </select>
                    </div>
                    <div class="input-group">
                        <label for="diff-model">Diffusion Model:</label>
                        <select id="diff-model" class="param-input">
                            <option value="" selected>keep current</option>
                            <option value="none">None</option>
                        </select>
                    </div>
                    <div class="line">
                        <div class="input-group">
                            <label for="clip_l">Clip_L:</label>
                            <select id="clip_l" class="param-input">
                                <option value="" selected>keep current</option>
                                <option value="none">None</option>
                            </select>
                        </div>
                        <div class="input-group">
                            <label for="clip_g">Clip_G:</label>
                            <select id="clip_g" class="param-input">
                                <option value="" selected>keep current</option>
                                <option value="none">None</option>
                            </select>
                        </div>
                        <div class="input-group">
                            <label for="t5xxl">T5 XXL:</label>
                            <select id="t5xxl" class="param-input">
                                <option value="" selected>keep current</option>
                                <option value="none">None</option>
                            </select>
                        </div>
                    </div>
                    <div class="line">
                        <div class="input-group">
                            <label for="vae">VAE:</label>
                            <select id="vae" class="param-input">
                                <option value="" selected>keep current</option>
                                <option value="none">None</option>
                            </select>
                        </div>
                        <div class="input-group">
                            <label for="tae">TAE:</label>
                            <select id="tae" class="param-input">
                                <option value="" selected>keep current</option>
                                <option value="none">None</option>
                            </select>
                        </div>
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
        let queued_tasks = 0;
        async function update_queue() {
            const display = document.getElementById('queued_tasks');
            display.innerHTML = queued_tasks;
        }
        const modelIdElement = document.getElementById('model-id');
        async function fetchModelId() {
            const response = await fetch('/model');
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
            const response = await fetch('/sample_methods');
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
            const response = await fetch('/schedules');
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
            const response = await fetch('/previews');
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
        async function fetchModelsEncodersAE() {
            const response = await fetch('/models');
            const data = await response.json();
            const modelsSelect = document.getElementById('model');
            if (data.models.length > 0) {
                data.models.forEach(model => {
                    const option = document.createElement('option');
                    option.value = model;
                    option.textContent = model;
                    modelsSelect.appendChild(option);
                });
            } else {
                modelsSelect.options.length = 1;
                const currentOption = modelsSelect.options[0];
                currentOption.select = true;
                currentOption.value = "";
                currentOption.textContent = "unavailable";
            }
            const diffModelsSelect = document.getElementById('diff-model');
            if (data.diffusion_models.length > 0) {
                data.diffusion_models.forEach(model => {
                    const option = document.createElement('option');
                    option.value = model;
                    option.textContent = model;
                    diffModelsSelect.appendChild(option);
                });
            } else {
                diffModelsSelect.options.length = 1;
                const currentOption = diffModelsSelect.options[0];
                currentOption.select = true;
                currentOption.value = "";
                currentOption.textContent = "unavailable";
            }
            const clipLSelect = document.getElementById('clip_l');
            if (data.text_encoders.length > 0) {
                data.text_encoders.forEach(encoder => {
                    const option = document.createElement('option');
                    option.value = encoder;
                    option.textContent = encoder;
                    clipLSelect.appendChild(option);
                });
            } else {
                clipLSelect.options.length = 1;
                const currentOption = clipLSelect.options[0];
                currentOption.select = true;
                currentOption.value = "";
                currentOption.textContent = "unavailable";
            }
            const clipGSelect = document.getElementById('clip_g');
            if (data.text_encoders.length > 0) {
                data.text_encoders.forEach(encoder => {
                    const option = document.createElement('option');
                    option.value = encoder;
                    option.textContent = encoder;
                    clipGSelect.appendChild(option);
                });
            } else {
                clipGSelect.options.length = 1;
                const currentOption = clipGSelect.options[0];
                currentOption.select = true;
                currentOption.value = "";
                currentOption.textContent = "unavailable";
            }
            const t5xxlSelect = document.getElementById('t5xxl');
            if (data.text_encoders.length > 0) {
                data.text_encoders.forEach(encoder => {
                    const option = document.createElement('option');
                    option.value = encoder;
                    option.textContent = encoder;
                    t5xxlSelect.appendChild(option);
                });
            } else {
                t5xxlSelect.options.length = 1;
                const currentOption = t5xxlSelect.options[0];
                currentOption.select = true;
                currentOption.value = "";
                currentOption.textContent = "unavailable";
            }
            const vaeSelect = document.getElementById('vae');
            if (data.vaes.length > 0) {
                data.vaes.forEach(ae => {
                    const option = document.createElement('option');
                    option.value = ae;
                    option.textContent = ae;
                    vaeSelect.appendChild(option);
                });
            } else {
                vaeSelect.options.length = 1;
                const currentOption = vaeSelect.options[0];
                currentOption.select = true;
                currentOption.value = "";
                currentOption.textContent = "unavailable";
            }
            const taeSelect = document.getElementById('tae');
            if (data.taes.length > 0) {
                data.taes.forEach(ae => {
                    const option = document.createElement('option');
                    option.value = ae;
                    option.textContent = ae;
                    taeSelect.appendChild(option);
                });
            } else {
                taeSelect.options.length = 1;
                const currentOption = taeSelect.options[0];
                currentOption.select = true;
                currentOption.value = "";
                currentOption.textContent = "unavailable";
            }
        }
        async function fetchParams() {
            const response = await fetch('/params');
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
            document.getElementById('vae_on_cpu').checked = data.context_params.vae_on_cpu;
            document.getElementById('clip_on_cpu').checked = data.context_params.clip_on_cpu;
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
        fetchModelsEncodersAE();
        fetchModelId();
        fetchParams();
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
            const vae_on_cpu = document.getElementById('vae_on_cpu').checked;
            const clip_on_cpu = document.getElementById('clip_on_cpu').checked;
            const vae_tiling = document.getElementById('vae_tiling').checked;
            const tae_decode = document.getElementById('tae_decode').checked;
            const preview_mode = document.getElementById('preview_mode').value;
            const preview_interval = document.getElementById('preview_interval').value;
            const canvas = document.getElementById('imageCanvas');
            const ctx = canvas.getContext('2d');
            const downloadLink = document.getElementById('downloadLink');
            const requestBody = {
                prompt: prompt,
                negative_prompt: neg_prompt,
                ...(width && { width: parseInt(width) }),
                ...(height && { height: parseInt(height) }),
                ...(cfg_scale && { cfg_scale: parseFloat(cfg_scale) }),
                ...(steps && { sample_steps: parseInt(steps) }),
                ...(guidance && { guidance: parseFloat(guidance) }),
                ...(sample_method && { sample_method: sample_method }),
                ...(seed && { seed: parseInt(seed) }),
                ...(batch_count && { batch_count: parseInt(batch_count) }),
                ...(schedule_method && { schedule: schedule_method }),
                ...(model && { model: model }),
                ...(diff_model && { diffusion_model: diff_model }),
                ...(clip_l && { clip_l: clip_l }),
                ...(clip_g && { clip_g: clip_g }),
                ...(t5xxl && { t5xxl: t5xxl }),
                ...(vae && { vae: vae }),
                ...(tae && { tae: tae }),
                ... { vae_on_cpu: vae_on_cpu },
                ... { clip_on_cpu: clip_on_cpu },
                ... { vae_tiling: vae_tiling },
                ... { tae_decode: tae_decode },
                ...(preview_mode && { preview_mode: preview_mode }),
                ...(preview_interval && { preview_interval: preview_interval }),
            };
            const response = await fetch('/txt2img', {
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
                const statusResponse = await fetch(`/result?task_id=${taskId}`);
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