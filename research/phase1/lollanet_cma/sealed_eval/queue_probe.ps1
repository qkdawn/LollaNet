param(
    [Parameter(Mandatory = $true)][int]$UpstreamPid,
    [Parameter(Mandatory = $true)][string]$Python,
    [Parameter(Mandatory = $true)][string]$DevelopmentManifest,
    [Parameter(Mandatory = $true)][string]$VectorTrainView,
    [Parameter(Mandatory = $true)][string]$DevelopmentView,
    [Parameter(Mandatory = $true)][string]$Output
)

$ErrorActionPreference = 'Stop'
Write-Output "PROBE_QUEUE_V1 waiting_for_pid=$UpstreamPid"
Wait-Process -Id $UpstreamPid
if (-not (Test-Path -LiteralPath $DevelopmentManifest)) {
    throw 'development queue ended without run_manifest.json; probe preparation stopped'
}
$development = Get-Content -Raw -LiteralPath $DevelopmentManifest | ConvertFrom-Json
if ($development.status -ne 'frozen' -or -not $development.hard_gate_pass) {
    throw "development did not freeze a passing selection; probe preparation stopped: status=$($development.status)"
}
Write-Output 'PROBE_QUEUE_V1 preparing_probe'
& $Python -m research.phase1.lollanet_cma.sealed_eval.prepare_probe `
    --development-manifest $DevelopmentManifest `
    --vector-train-view $VectorTrainView `
    --development-view $DevelopmentView `
    --output $Output
if ($LASTEXITCODE -ne 0) {
    throw "probe preparation failed with exit code $LASTEXITCODE"
}
Write-Output 'PROBE_QUEUE_V1 complete=probe_frozen'
