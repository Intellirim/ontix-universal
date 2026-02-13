$base = "C:\Users\hanso\ontix-universal"
$dist = Join-Path $base "dist"
if (!(Test-Path $dist)) { New-Item -ItemType Directory -Path $dist | Out-Null }

$packs = @("beauty","fnb","fashion","tech-saas","fitness","entertainment")
foreach ($pack in $packs) {
    $src = Join-Path $base "packs\$pack"
    $dst = Join-Path $dist "ontix-$pack-pack.zip"
    if (Test-Path $dst) { Remove-Item $dst }
    Compress-Archive -Path (Join-Path $src "*") -DestinationPath $dst -Force
    Write-Host "Created: ontix-$pack-pack.zip"
}

# Bundle: all 6 packs in one ZIP (preserving directory structure)
$bundleDst = Join-Path $dist "ontix-all-packs-bundle.zip"
if (Test-Path $bundleDst) { Remove-Item $bundleDst }
$packPaths = $packs | ForEach-Object { Join-Path $base "packs\$_" }
Compress-Archive -Path $packPaths -DestinationPath $bundleDst -Force
Write-Host "Created: ontix-all-packs-bundle.zip"

Write-Host ""
Write-Host "=== All ZIP files ==="
Get-ChildItem $dist -Filter "*.zip" | ForEach-Object {
    $sizeKB = [math]::Round($_.Length / 1KB, 1)
    Write-Host "$($_.Name) - ${sizeKB} KB"
}
