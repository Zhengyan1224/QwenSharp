param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$ModelUrl = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/matcha-icefall-zh-en.tar.bz2"
$VocosUrl = "https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos-16khz-univ.onnx"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path -LiteralPath (Join-Path $ScriptDir "..")).Path
$DestRoot = Join-Path $RepoRoot "models\tts"
$ModelDir = Join-Path $DestRoot "matcha-icefall-zh-en"
$TempDir = Join-Path ([System.IO.Path]::GetTempPath()) ("qwensharp-tts-" + [Guid]::NewGuid().ToString("N"))
$ArchivePath = Join-Path $TempDir "matcha-icefall-zh-en.tar.bz2"
$VocosPath = Join-Path $ModelDir "vocos-16khz-univ.onnx"

function Download-File {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [Parameter(Mandatory = $true)][string]$OutputPath
    )

    $parent = Split-Path -Parent $OutputPath
    if (-not [string]::IsNullOrWhiteSpace($parent)) {
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
    }

    Invoke-WebRequest -Uri $Url -OutFile $OutputPath -UseBasicParsing
}

function Test-CoreModelFiles {
    return (Test-Path -LiteralPath (Join-Path $ModelDir "model-steps-3.onnx")) `
        -and (Test-Path -LiteralPath (Join-Path $ModelDir "tokens.txt")) `
        -and (Test-Path -LiteralPath (Join-Path $ModelDir "lexicon.txt")) `
        -and (Test-Path -LiteralPath (Join-Path $ModelDir "espeak-ng-data"))
}

function Get-TarCandidates {
    $candidates = New-Object System.Collections.Generic.List[string]

    if ($env:WINDIR) {
        $systemTar = Join-Path $env:WINDIR "System32\tar.exe"
        if (Test-Path -LiteralPath $systemTar) {
            $candidates.Add($systemTar)
        }
    }

    $pathTar = Get-Command tar -ErrorAction SilentlyContinue
    if ($pathTar) {
        $candidates.Add($pathTar.Source)
    }

    return $candidates | Select-Object -Unique
}

function Invoke-TarExtract {
    param(
        [Parameter(Mandatory = $true)][string]$TarPath,
        [Parameter(Mandatory = $true)][string]$Archive,
        [Parameter(Mandatory = $true)][string]$Destination
    )

    $attempts = @(
        @("-xjf", $Archive, "-C", $Destination),
        @("-xf", $Archive, "-C", $Destination)
    )

    foreach ($tarArgs in $attempts) {
        & $TarPath @tarArgs
        if ($LASTEXITCODE -eq 0) {
            return $true
        }
    }

    return $false
}

function Expand-TarBz2WithDotNet {
    param(
        [Parameter(Mandatory = $true)][string]$Archive,
        [Parameter(Mandatory = $true)][string]$Destination
    )

    $dotnet = Get-Command dotnet -ErrorAction SilentlyContinue
    if (-not $dotnet) {
        throw "Unable to extract .tar.bz2. Install bzip2-capable tar, 7-Zip, or the .NET SDK."
    }

    $extractorDir = Join-Path $TempDir "extractor"
    New-Item -ItemType Directory -Force -Path $extractorDir | Out-Null

    $csproj = @'
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net10.0</TargetFramework>
    <ImplicitUsings>enable</ImplicitUsings>
    <Nullable>enable</Nullable>
  </PropertyGroup>
  <ItemGroup>
    <PackageReference Include="SharpCompress" Version="0.39.0" />
  </ItemGroup>
</Project>
'@

    $program = @'
using SharpCompress.Common;
using SharpCompress.Readers;

if (args.Length != 2)
{
    Console.Error.WriteLine("Usage: extractor <archive.tar.bz2> <destination>");
    return 2;
}

var archivePath = Path.GetFullPath(args[0]);
var destination = Path.GetFullPath(args[1]);
Directory.CreateDirectory(destination);

using var stream = File.OpenRead(archivePath);
using var reader = ReaderFactory.Open(stream);

while (reader.MoveToNextEntry())
{
    if (reader.Entry.IsDirectory)
    {
        continue;
    }

    var key = (reader.Entry.Key ?? string.Empty).Replace('\\', '/');
    var parts = key.Split('/', StringSplitOptions.RemoveEmptyEntries);
    if (Path.IsPathRooted(key) || parts.Any(part => part == ".."))
    {
        throw new InvalidOperationException($"Unsafe archive entry path: {reader.Entry.Key}");
    }

    reader.WriteEntryToDirectory(destination, new ExtractionOptions
    {
        ExtractFullPath = true,
        Overwrite = true,
    });
}

return 0;
'@

    Set-Content -LiteralPath (Join-Path $extractorDir "QwenSharp.TtsExtractor.csproj") -Value $csproj -Encoding UTF8
    Set-Content -LiteralPath (Join-Path $extractorDir "Program.cs") -Value $program -Encoding UTF8

    & dotnet run --project (Join-Path $extractorDir "QwenSharp.TtsExtractor.csproj") -- $Archive $Destination
    if ($LASTEXITCODE -ne 0) {
        throw "The .NET fallback extractor failed with exit code $LASTEXITCODE."
    }
}

function Expand-TarBz2 {
    param(
        [Parameter(Mandatory = $true)][string]$Archive,
        [Parameter(Mandatory = $true)][string]$Destination
    )

    foreach ($tar in Get-TarCandidates) {
        Write-Host "Trying tar extractor: $tar"
        if (Invoke-TarExtract -TarPath $tar -Archive $Archive -Destination $Destination) {
            return
        }
    }

    Write-Warning "System tar could not extract the .tar.bz2 archive. Falling back to a temporary .NET extractor."
    Expand-TarBz2WithDotNet -Archive $Archive -Destination $Destination
}

New-Item -ItemType Directory -Force -Path $DestRoot | Out-Null
New-Item -ItemType Directory -Force -Path $TempDir | Out-Null

try {
    if ((Test-CoreModelFiles) -and -not $Force) {
        Write-Host "TTS model files already exist: $ModelDir"
    }
    else {
        Write-Host "Downloading matcha-icefall-zh-en..."
        Download-File -Url $ModelUrl -OutputPath $ArchivePath

        Write-Host "Extracting to $DestRoot..."
        Expand-TarBz2 -Archive $ArchivePath -Destination $DestRoot
    }

    New-Item -ItemType Directory -Force -Path $ModelDir | Out-Null
    if ((Test-Path -LiteralPath $VocosPath) -and -not $Force) {
        Write-Host "Vocoder already exists: $VocosPath"
    }
    else {
        Write-Host "Downloading vocos-16khz-univ.onnx..."
        Download-File -Url $VocosUrl -OutputPath $VocosPath
    }

    $requiredPaths = @(
        (Join-Path $ModelDir "model-steps-3.onnx"),
        (Join-Path $ModelDir "tokens.txt"),
        (Join-Path $ModelDir "lexicon.txt"),
        (Join-Path $ModelDir "espeak-ng-data"),
        (Join-Path $ModelDir "vocos-16khz-univ.onnx")
    )

    foreach ($required in $requiredPaths) {
        if (-not (Test-Path -LiteralPath $required)) {
            throw "Missing required TTS file after installation: $required"
        }
    }

    Write-Host "TTS model is ready: $ModelDir"
}
finally {
    if (Test-Path -LiteralPath $TempDir) {
        Remove-Item -LiteralPath $TempDir -Recurse -Force
    }
}
