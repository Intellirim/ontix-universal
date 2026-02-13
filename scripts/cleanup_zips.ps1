$dist = "C:\Users\hanso\ontix-universal\dist"
Remove-Item (Join-Path $dist "ontix-pack-*.zip") -Force -ErrorAction SilentlyContinue
Get-ChildItem $dist -Filter "*.zip" | ForEach-Object {
    $sizeKB = [math]::Round($_.Length / 1KB, 1)
    Write-Host "$($_.Name) - $sizeKB KB"
}
