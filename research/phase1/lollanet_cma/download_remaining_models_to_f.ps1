param(
    [string]$Root = 'F:\LollaNetData\lollanet-cma\qualification\checkpoints'
)

$ErrorActionPreference = 'Stop'
$env:GIT_LFS_SKIP_SMUDGE = '1'

function Download-Model {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Repository,
        [Parameter(Mandatory = $true)][string]$Revision
    )

    $target = Join-Path $Root $Name
    New-Item -ItemType Directory -Force -Path $Root | Out-Null
    if (Test-Path -LiteralPath $target) {
        if (-not (Test-Path -LiteralPath (Join-Path $target '.git'))) {
            if (Get-ChildItem -LiteralPath $target -Force -ErrorAction SilentlyContinue) {
                throw "Target exists but is not a resumable Git-LFS checkout: $target"
            }
            git clone --no-checkout $Repository $target
        }
    } else {
        git clone --no-checkout $Repository $target
    }

    Write-Output "DOWNLOAD_F_V1 checkout=$Name revision=$Revision"
    git -C $target fetch --depth=1 origin $Revision
    git -C $target checkout --detach $Revision
    git -C $target lfs pull
    $observed = (git -C $target rev-parse HEAD).Trim()
    if ($observed -ne $Revision) {
        throw "Revision mismatch for ${Name}: $observed"
    }
    Set-Content -LiteralPath (Join-Path $target 'REVISION') -Value ($Revision + "`n") -Encoding utf8NoBOM
    Write-Output "DOWNLOAD_F_V1 complete=$Name"
}

Download-Model -Name 'qwen3-8b' -Repository 'https://www.modelscope.cn/Qwen/Qwen3-8B.git' -Revision '26028140be3ee69b82b1d1450179ab71bb1121b9'
Download-Model -Name 'qwen2.5-3b-instruct' -Repository 'https://www.modelscope.cn/Qwen/Qwen2.5-3B-Instruct.git' -Revision '8f4992eda43eea7c770690ddc0de8f732da246f5'
Write-Output 'DOWNLOAD_F_V1 complete=all_remaining_models'
