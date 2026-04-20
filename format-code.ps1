param(
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$repoRoot = $PSScriptRoot
if (-not $repoRoot) {
    $repoRoot = (Get-Location).Path
}

$patterns = @(
    "src/*.cpp"
    "src/*.h"
    "src/*.hpp"
    "src/vocab/*.h"
    "src/vocab/*.cpp"
    "examples/cli/*.cpp"
    "examples/common/*.hpp"
    "examples/cli/*.h"
    "examples/server/*.cpp"
)

Push-Location $repoRoot
try {
    if (-not $DryRun) {
        $null = Get-Command clang-format -ErrorAction Stop
    }

    foreach ($pattern in $patterns) {
        $files = Get-ChildItem -Path $pattern -File -ErrorAction SilentlyContinue | Sort-Object FullName

        foreach ($file in $files) {
            $relativePath = $file.FullName.Substring($repoRoot.Length).TrimStart('\', '/') -replace '\\', '/'

            if ($relativePath -like "vocab*") {
                continue
            }

            Write-Host "formatting '$relativePath'"

            # if ($file.Name -ne "stable-diffusion.h") {
            #     clang-tidy -fix -p build_linux/ "$relativePath"
            # }

            if (-not $DryRun) {
                & clang-format -style=file -i $file.FullName
            }
        }
    }
}
finally {
    Pop-Location
}
