$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$rawRoot = Join-Path $projectRoot "data\raw"
$archive = Join-Path $rawRoot "trec-covid.zip"
$datasetDir = Join-Path $rawRoot "trec-covid"
$sourceUrl = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/trec-covid.zip"

New-Item -ItemType Directory -Path $rawRoot -Force | Out-Null
if (-not (Test-Path -LiteralPath $archive)) {
    Invoke-WebRequest -Uri $sourceUrl -OutFile $archive -UseBasicParsing
}
if (-not (Test-Path -LiteralPath $datasetDir)) {
    Expand-Archive -LiteralPath $archive -DestinationPath $rawRoot
}

Write-Host "TREC-COVID is ready at $datasetDir"
