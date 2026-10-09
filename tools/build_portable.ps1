# Build the portable SmartFitter app:  .\tools\build_portable.ps1
# Output: dist\SmartFitter\SmartFitter.exe and dist\SmartFitter-portable-<version>.zip
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $py)) { throw "Missing $py - create the venv first." }

& $py -m pip install --quiet pyinstaller
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$version = (& $py -c "from smartfitter import core; print(core.APP_VERSION)").Trim()
$sw = [Diagnostics.Stopwatch]::StartNew()
& $py -m PyInstaller --noconfirm --clean --workpath build\pyinstaller --distpath dist tools\smartfitter.spec
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$zip = "dist\SmartFitter-portable-$version.zip"
if (Test-Path $zip) { Remove-Item $zip }
Compress-Archive -Path dist\SmartFitter -DestinationPath $zip
$mb = [math]::Round(((Get-ChildItem dist\SmartFitter -Recurse -File | Measure-Object Length -Sum).Sum) / 1MB, 1)
$zmb = [math]::Round((Get-Item $zip).Length / 1MB, 1)
Write-Host ("Built dist\SmartFitter ({0} MB), {1} ({2} MB) in {3:n0} s" -f $mb, $zip, $zmb, $sw.Elapsed.TotalSeconds)
