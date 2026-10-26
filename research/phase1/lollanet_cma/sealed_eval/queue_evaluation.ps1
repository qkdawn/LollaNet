param(
    [Parameter(Mandatory = $true)][int]$ReleasePid,
    [Parameter(Mandatory = $true)][int]$ProbePid,
    [Parameter(Mandatory = $true)][string]$Python,
    [Parameter(Mandatory = $true)][string]$EvaluationRelease,
    [Parameter(Mandatory = $true)][string]$ProbeManifest,
    [Parameter(Mandatory = $true)][string]$Preflight,
    [Parameter(Mandatory = $true)][string]$RawOutput,
    [Parameter(Mandatory = $true)][string]$ReportOutput
)

$ErrorActionPreference = 'Stop'
Write-Output "SEALED_EVALUATION_QUEUE_V1 waiting_for_release_pid=$ReleasePid probe_pid=$ProbePid"
Wait-Process -Id $ReleasePid
Wait-Process -Id $ProbePid
if (-not (Test-Path -LiteralPath $EvaluationRelease)) {
    throw 'release queue ended without evaluation-release manifest; sealed evaluation stopped'
}
if (-not (Test-Path -LiteralPath $ProbeManifest)) {
    throw 'probe queue ended without run_manifest.json; sealed evaluation stopped'
}
Write-Output 'SEALED_EVALUATION_QUEUE_V1 auditing_preflight'
& $Python -m research.phase1.lollanet_cma.sealed_eval.audit_preflight `
    --evaluation-release $EvaluationRelease `
    --probe-manifest $ProbeManifest `
    --output $Preflight
if ($LASTEXITCODE -ne 0) {
    throw "sealed evaluation preflight failed with exit code $LASTEXITCODE"
}
$audit = Get-Content -Raw -LiteralPath $Preflight | ConvertFrom-Json
if ($audit.sealed_evaluation_gate_status -ne 'ready' -or -not $audit.evaluation_authorized) {
    throw "sealed evaluation preflight is not ready: $($audit.sealed_evaluation_gate_status)"
}
Write-Output 'SEALED_EVALUATION_QUEUE_V1 running_raw_conditions'
& $Python -m research.phase1.lollanet_cma.sealed_eval.run_evaluation `
    --preflight $Preflight `
    --output $RawOutput
if ($LASTEXITCODE -ne 0) {
    throw "sealed raw evaluation failed with exit code $LASTEXITCODE"
}
Write-Output 'SEALED_EVALUATION_QUEUE_V1 assembling_report'
& $Python -m research.phase1.lollanet_cma.sealed_eval.assemble_report `
    --raw-manifest "$RawOutput\run_manifest.json" `
    --output $ReportOutput
if ($LASTEXITCODE -ne 0) {
    throw "sealed report assembly failed with exit code $LASTEXITCODE"
}
Write-Output 'SEALED_EVALUATION_QUEUE_V1 complete=report_assembled'
