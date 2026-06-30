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
    "src/extensions/*.cpp"
    "src/extensions/*.h"
    "src/extensions/*.hpp"
    "src/runtime/*.cpp"
    "src/runtime/*.h"
    "src/runtime/*.hpp"
    "src/model/*/*.cpp"
    "src/model/*/*.h"
    "src/model/*/*.hpp"
    "src/tokenizers/*.h"
    "src/tokenizers/*.cpp"
    "src/tokenizers/vocab/*.h"
    "src/tokenizers/vocab/*.cpp"
    "src/model_io/*.h"
    "src/model_io/*.cpp"
    "examples/cli/*.cpp"
    "examples/cli/*.h"
    "examples/server/*.cpp"
    "examples/common/*.hpp"
    "examples/common/*.h"
    "examples/common/*.cpp"
)

$root = (Get-Location).Path

foreach ($pattern in $patterns) {
    $files = Get-ChildItem -Path $pattern -File -ErrorAction SilentlyContinue | Sort-Object FullName

    foreach ($file in $files) {
        $relativePath = $file.FullName.Substring($root.Length).TrimStart('\', '/') -replace '\\', '/'

        if ($relativePath -like "vocab*") {
            continue
        }

        Write-Host "formatting '$relativePath'"

        # if ($relativePath -ne "stable-diffusion.h") {
        #     clang-tidy -fix -p build_linux/ "$relativePath"
        # }

        & clang-format -style=file -i $relativePath
    }
}
