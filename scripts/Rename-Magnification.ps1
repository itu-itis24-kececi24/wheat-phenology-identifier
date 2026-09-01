param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Folder,

    [Parameter(Position = 1)]
    [ValidateSet("10X_TO_1X", "1X_TO_10X")]
    [string]$Mode = "10X_TO_1X"
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

    if ($Mode -eq "10X_TO_1X") {
        $newBase = $base -replace '(^|[-_])10x($|[-_])', '${1}1X${2}'
    }
    else {
        $newBase = $base -replace '(^|[-_])1x($|[-_])', '${1}10X${2}'
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