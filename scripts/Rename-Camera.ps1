param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Folder,

    [Parameter(Position = 1)]
    [ValidateSet("K1_TO_K2", "K2_TO_K1")]
    [string]$Mode = "K1_TO_K2"
)

# Resolve and validate folder
if (-not (Test-Path -LiteralPath $Folder -PathType Container)) {
    Write-Error "Folder does not exist: $Folder"
    exit 1
}

$Folder = (Resolve-Path -LiteralPath $Folder).Path

Write-Host ""
Write-Host "Folder: $Folder"
Write-Host "Mode:   $Mode"
Write-Host ""

# Build a list of proposed changes first
$changes = @()

Get-ChildItem -LiteralPath $Folder -File | ForEach-Object {
    $base = $_.BaseName
    $ext  = $_.Extension

    if ($Mode -eq "K1_TO_K2") {
        $newBase = $base -replace '(^|[-_])K1($|[-_])', '${1}K2${2}'
    }
    else {
        $newBase = $base -replace '(^|[-_])K2($|[-_])', '${1}K1${2}'
    }

    if ($newBase -ne $base) {
        $changes += [PSCustomObject]@{
            File    = $_
            OldName = $_.Name
            NewName = $newBase + $ext
        }
    }
}

if ($changes.Count -eq 0) {
    Write-Host "No matching files found."
    exit 0
}

# Preview
Write-Host "Proposed changes:"
Write-Host "-----------------"

foreach ($change in $changes) {
    Write-Host "$($change.OldName)"
    Write-Host "  -> $($change.NewName)"
}

Write-Host ""
Write-Host "$($changes.Count) file(s) will be renamed."
Write-Host ""
Write-Host "Press ENTER to apply these changes."
Write-Host "Press Ctrl+C to cancel."

Read-Host | Out-Null

# Apply
foreach ($change in $changes) {
    Rename-Item `
        -LiteralPath $change.File.FullName `
        -NewName $change.NewName
}

Write-Host ""
Write-Host "Done. Renamed $($changes.Count) file(s)."