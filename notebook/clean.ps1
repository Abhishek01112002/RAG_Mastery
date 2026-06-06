$json = Get-Content -Raw -Path 'e:\RAG\notebook\multimodal-rag-pdf-with-images.ipynb' | ConvertFrom-Json
foreach ($cell in $json.cells) {
    if ($cell.attachments) { $cell.attachments = $null }
    if ($cell.outputs) { $cell.outputs = @() }
}
$json | ConvertTo-Json -Depth 100 | Out-File -FilePath 'e:\RAG\notebook\temp_clean.ipynb' -Encoding utf8
