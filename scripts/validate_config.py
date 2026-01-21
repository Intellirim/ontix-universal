### **scripts/validate_config.py**
#!/usr/bin/env python3
"""
Config Validation Script
브랜드 설정 검증
"""

import typer
import yaml
from pathlib import Path
from rich.console import Console
from rich.table import Table
from typing import Optional

app = typer.Typer()
console = Console()


@app.command()
def validate(
    brand_id: Optional[str] = None,
    verbose: bool = False
):
    """
    브랜드 설정 검증
    
    Args:
        brand_id: 브랜드 ID (없으면 전체 검증)
        verbose: 상세 출력
    """
    console.print("\n🔍 [bold cyan]ONTIX Universal - Config Validator[/bold cyan]\n")
    
    if brand_id:
        brands = [brand_id]
    else:
        # 모든 브랜드 찾기
        brands_dir = Path("configs/brands")
        brands = [
            f.stem for f in brands_dir.glob("*.yaml")
            if not f.stem.startswith("_")
        ]
    
    if not brands:
        console.print("[red]❌ 검증할 브랜드가 없습니다.[/red]")
        return
    
    # 검증 실행
    results = []
    
    for brand in brands:
        result = _validate_brand(brand, verbose)
        results.append(result)
    
    # 결과 출력
    _print_results(results)


def _validate_brand(brand_id: str, verbose: bool) -> dict:
    """브랜드 검증"""
    config_path = Path(f"configs/brands/{brand_id}.yaml")
    
    result = {
        'brand_id': brand_id,
        'exists': config_path.exists(),
        'valid': False,
        'errors': [],
        'warnings': []
    }
    
    if not result['exists']:
        result['errors'].append("Config file not found")
        return result
    
    # YAML 로드
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        result['errors'].append(f"YAML parse error: {e}")
        return result
    
    # 필수 필드 검증
    required_fields = {
        'brand': ['id', 'name'],
        'features': None,
        'neo4j': ['brand_id'],
        'retrieval': None,
        'generation': None
    }
    
    for section, fields in required_fields.items():
        if section not in config:
            result['errors'].append(f"Missing section: {section}")
            continue
        
        if fields:
            for field in fields:
                if field not in config[section]:
                    result['errors'].append(f"Missing {section}.{field}")
    
    # 브랜드 ID 일치 확인
    if config.get('brand', {}).get('id') != brand_id:
        result['errors'].append(
            f"Brand ID mismatch: config={config['brand']['id']}, file={brand_id}"
        )
    
    # Features 검증
    features = config.get('features', [])
    
    valid_features = [
        'conversational', 'factual', 'product_recommendation',
        'analytics', 'advisor', 'content_generation',
        'social_monitoring', 'onboarding'
    ]
    
    for feature in features:
        if feature not in valid_features:
            result['warnings'].append(f"Unknown feature: {feature}")
    
    # Retrieval 설정 검증
    retrieval = config.get('retrieval', {})
    
    for feature in features:
        if feature not in retrieval:
            result['warnings'].append(f"No retrieval config for: {feature}")
    
    # Generation 설정 검증
    generation = config.get('generation', {})
    
    for feature in features:
        if feature not in generation:
            result['warnings'].append(f"No generation config for: {feature}")
        else:
            # 프롬프트 파일 존재 확인
            gen_config = generation[feature]
            
            prompt = gen_config.get('prompt')
            fallback = gen_config.get('fallback_prompt')
            
            if prompt:
                prompt_path = Path(f"prompts/{prompt}")
                if not prompt_path.exists():
                    result['warnings'].append(f"Prompt not found: {prompt}")
            
            if fallback:
                fallback_path = Path(f"prompts/{fallback}")
                if not fallback_path.exists():
                    result['errors'].append(f"Fallback prompt not found: {fallback}")
    
    # 검증 완료
    result['valid'] = len(result['errors']) == 0
    
    if verbose and result['valid']:
        console.print(f"[green]✅ {brand_id}: Valid[/green]")
    elif verbose and not result['valid']:
        console.print(f"[red]❌ {brand_id}: Invalid[/red]")
        for error in result['errors']:
            console.print(f"  [red]ERROR: {error}[/red]")
        for warning in result['warnings']:
            console.print(f"  [yellow]WARN: {warning}[/yellow]")
    
    return result


def _print_results(results: list):
    """결과 출력"""
    table = Table(title="Validation Results")
    
    table.add_column("Brand ID", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Errors", style="red")
    table.add_column("Warnings", style="yellow")
    
    for result in results:
        if not result['exists']:
            table.add_row(
                result['brand_id'],
                "❌ Not Found",
                "1",
                "0"
            )
        elif result['valid']:
            table.add_row(
                result['brand_id'],
                "✅ Valid",
                "0",
                str(len(result['warnings']))
            )
        else:
            table.add_row(
                result['brand_id'],
                "❌ Invalid",
                str(len(result['errors'])),
                str(len(result['warnings']))
            )
    
    console.print(table)
    
    # 상세 오류
    console.print("\n[bold]Details:[/bold]\n")
    
    for result in results:
        if result['errors']:
            console.print(f"[red]❌ {result['brand_id']}:[/red]")
            for error in result['errors']:
                console.print(f"  • {error}")
            console.print()
        
        if result['warnings']:
            console.print(f"[yellow]⚠️  {result['brand_id']}:[/yellow]")
            for warning in result['warnings']:
                console.print(f"  • {warning}")
            console.print()


if __name__ == "__main__":
    app()
