$patterns = @(
    "src/*.cpp"
    "src/*.h"
    "src/*.hpp"
    "src/conditioning/*.cpp"
    "src/conditioning/*.h"
    "src/conditioning/*.hpp"
    "src/core/*.cpp"
    "src/core/*.h"
    "src/core/*.hpp"
    "src/core/cpu/*.cpp"
    "src/core/cpu/*.h"
    "src/core/cuda/*.cu"
    "src/core/cuda/*.cuh"
    "src/core/cuda/*.h"
    "src/core/cuda/sageattention/*.cuh"
    "src/core/probe/*.cpp"
    "src/core/probe/*.h"
    "src/core/vulkan/*.cpp"
    "src/core/vulkan/*.h"
    "src/core/vulkan/shaders/*.comp"
    "src/extensions/*.cpp"
    "src/extensions/*.h"
    "src/extensions/*.hpp"
    "src/pipeline/*.cpp"
    "src/pipeline/*.h"
    "src/runtime/*.cpp"
    "src/runtime/*.h"
    "src/runtime/*.hpp"
    "src/model/*/*.cpp"
    "src/model/*/*.h"
    "src/model/*/*.hpp"
    "src/tokenizers/*.h"
    "src/tokenizers/*.cpp"
    "src/model_io/*.h"
    "src/model_io/*.cpp"
    "examples/cli/*.cpp"
    "examples/cli/*.h"
    "examples/server/*.cpp"
    "examples/common/*.hpp"
    "examples/common/*.h"
    "examples/common/*.cpp"
)

Push-Location (Join-Path $PSScriptRoot "..")

try {
    $root = (Get-Location).Path

    foreach ($pattern in $patterns) {
        $files = Get-ChildItem -Path $pattern -File -ErrorAction SilentlyContinue | Sort-Object FullName

        foreach ($file in $files) {
            $relativePath = $file.FullName.Substring($root.Length).TrimStart('\', '/') -replace '\\', '/'

            Write-Host "formatting '$relativePath'"

            # if ($relativePath -ne "stable-diffusion.h") {
            #     clang-tidy -fix -p build_linux/ "$relativePath"
            # }

            & clang-format -style=file -i $relativePath
        }
    }
} finally {
    Pop-Location
}
