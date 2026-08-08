param(
    [string]$Version = '1.0.0'
)

$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $root '.venv-build\Scripts\python.exe'
if (-not (Test-Path $venvPython)) {
    throw ".venv-build is missing. Create it first."
}

$isccCandidates = @()
$pathCommand = Get-Command iscc.exe -ErrorAction SilentlyContinue
if ($pathCommand) {
    $isccCandidates += $pathCommand.Source
}
$isccCandidates += @(
    (Join-Path $env:LOCALAPPDATA 'Programs\Inno Setup 6\ISCC.exe'),
    'C:\Program Files (x86)\Inno Setup 6\ISCC.exe',
    'C:\Program Files\Inno Setup 6\ISCC.exe'
)
$iscc = $isccCandidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
if (-not $iscc) {
    throw 'ISCC.exe not found. Install Inno Setup 6 first.'
}

$buildDir = Join-Path $root 'build'
$distDir = Join-Path $root 'dist'
$installerOutputDir = Join-Path $root 'installer\Output'
$installerExe = Join-Path $installerOutputDir ("DXF-Extract-Labels-Setup-$Version.exe")
$zipPath = Join-Path $root ("DXF-Extract-Labels-Setup-$Version.zip")
Remove-Item $buildDir, $distDir, $installerOutputDir -Recurse -Force -ErrorAction SilentlyContinue
if (Test-Path $zipPath) {
    Remove-Item $zipPath -Force
}

Write-Host 'Building PyInstaller bundle...'
& $venvPython -m PyInstaller --noconfirm --clean (Join-Path $root 'run_app.spec')
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller build failed with exit code $LASTEXITCODE"
}

Write-Host 'Building installer...'
& $iscc "/DMyAppVersion=$Version" (Join-Path $root 'installer\DXF-Extract-Labels.iss')
if ($LASTEXITCODE -ne 0) {
    throw "Inno Setup build failed with exit code $LASTEXITCODE"
}

if (-not (Test-Path $installerExe)) {
    throw "Installer was not created: $installerExe"
}

Write-Host 'Building zip package...'
Compress-Archive -LiteralPath $installerExe -DestinationPath $zipPath -CompressionLevel Optimal

Write-Host "Build completed: $installerExe"
Write-Host "Zip completed: $zipPath"
