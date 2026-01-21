### **scripts/create_brand.py**
#!/usr/bin/env python3
"""
Brand Creation Script
새로운 브랜드 설정 생성
"""

import typer
import yaml
from pathlib import Path
from rich.console import Console
from rich.prompt import Prompt, Confirm
from typing import Optional

app = typer.Typer()
console = Console()


@app.command()
def create(
    brand_id: Optional[str] = None,
    interactive: bool = True
):
    """
    새로운 브랜드 생성
    
    Args:
        brand_id: 브랜드 ID
        interactive: 대화형 모드
    """
    console.print("\n🎨 [bold cyan]ONTIX Universal - Brand Creator[/bold cyan]\n")
    
    # 브랜드 ID 입력
    if not brand_id:
        brand_id = Prompt.ask("브랜드 ID를 입력하세요 (영문 소문자, 하이픈 가능)")
    
    brand_id = brand_id.lower().strip()
    
    # 파일 존재 확인
    config_path = Path(f"configs/brands/{brand_id}.yaml")
    
    if config_path.exists():
        console.print(f"[red]❌ 브랜드가 이미 존재합니다: {brand_id}[/red]")
        
        if not Confirm.ask("덮어쓰시겠습니까?"):
            console.print("[yellow]취소되었습니다.[/yellow]")
            return
    
    # 대화형 입력
    if interactive:
        brand_data = _interactive_input(brand_id)
    else:
        brand_data = _create_minimal_config(brand_id)
    
    # YAML 저장
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(brand_data, f, allow_unicode=True, sort_keys=False)
    
    console.print(f"\n[green]✅ 브랜드 생성 완료: {config_path}[/green]")
    
    # 프롬프트 폴더 생성 여부
    if Confirm.ask("\n브랜드 전용 프롬프트 폴더를 생성하시겠습니까?"):
        _create_prompt_folder(brand_id)
    
    console.print("\n[bold green]🎉 완료![/bold green]")
    console.print(f"\n다음 단계:")
    console.print(f"1. configs/brands/{brand_id}.yaml 설정 확인")
    console.print(f"2. python scripts/validate_config.py {brand_id}")
    console.print(f"3. python scripts/sync_brand.py {brand_id}")


def _interactive_input(brand_id: str) -> dict:
    """대화형 입력"""
    console.print("\n[cyan]브랜드 정보를 입력하세요:[/cyan]\n")
    
    brand_name = Prompt.ask("브랜드 이름", default=brand_id.upper())
    description = Prompt.ask("브랜드 설명")
    industry = Prompt.ask("산업 분류", default="General")
    
    # 기능 선택
    console.print("\n[cyan]활성화할 기능을 선택하세요:[/cyan]")
    
    all_features = [
        "conversational",
        "factual",
        "product_recommendation",
        "analytics",
        "advisor",
        "content_generation",
        "social_monitoring",
        "onboarding"
    ]
    
    features = []
    
    for feature in all_features:
        if Confirm.ask(f"  {feature}", default=feature in ["conversational", "factual"]):
            features.append(feature)
    
    # 설정 생성
    config = {
        'brand': {
            'id': brand_id,
            'name': brand_name,
            'description': description,
            'industry': industry
        },
        'features': features,
        'neo4j': {
            'brand_id': brand_id,
            'namespaces': [brand_id],
            'vector_index': 'ontix_global_concept_index'
        },
        'retrieval': _create_retrieval_config(features),
        'generation': _create_generation_config(features)
    }
    
    return config


def _create_minimal_config(brand_id: str) -> dict:
    """최소 설정 생성"""
    return {
        'brand': {
            'id': brand_id,
            'name': brand_id.upper(),
            'description': f'{brand_id} brand',
            'industry': 'General'
        },
        'features': ['conversational', 'factual'],
        'neo4j': {
            'brand_id': brand_id,
            'namespaces': [brand_id],
            'vector_index': 'ontix_global_concept_index'
        },
        'retrieval': {
            'factual': {
                'retrievers': ['graph', 'vector'],
                'max_results': 10
            },
            'conversational': {
                'retrievers': ['graph'],
                'max_results': 5
            }
        },
        'generation': {
            'factual': {
                'type': 'factual',
                'fallback_prompt': 'shared/factual/base.txt',
                'model': 'mini',
                'temperature': 0
            },
            'conversational': {
                'type': 'conversational',
                'fallback_prompt': 'shared/conversational/base.txt',
                'model': 'full',
                'temperature': 0.8
            }
        }
    }


def _create_retrieval_config(features: list) -> dict:
    """Retrieval 설정 생성"""
    config = {}
    
    retrieval_templates = {
        'factual': {
            'retrievers': ['graph', 'vector'],
            'max_results': 10
        },
        'product_recommendation': {
            'retrievers': ['product', 'vector'],
            'max_results': 20
        },
        'analytics': {
            'retrievers': ['stats', 'graph'],
            'max_results': 50
        },
        'advisor': {
            'retrievers': ['vector', 'graph'],
            'max_results': 10
        },
        'conversational': {
            'retrievers': ['graph'],
            'max_results': 5
        }
    }
    
    for feature in features:
        if feature in retrieval_templates:
            config[feature] = retrieval_templates[feature]
    
    return config


def _create_generation_config(features: list) -> dict:
    """Generation 설정 생성"""
    config = {}
    
    generation_templates = {
        'factual': {
            'type': 'factual',
            'fallback_prompt': 'shared/factual/base.txt',
            'model': 'mini',
            'temperature': 0
        },
        'product_recommendation': {
            'type': 'recommendation',
            'fallback_prompt': 'shared/factual/product.txt',
            'model': 'full',
            'temperature': 0.7
        },
        'analytics': {
            'type': 'insight',
            'fallback_prompt': 'shared/insight/base.txt',
            'model': 'full',
            'temperature': 0.7
        },
        'advisor': {
            'type': 'insight',
            'fallback_prompt': 'shared/insight/advisor.txt',
            'model': 'full',
            'temperature': 0.7
        },
        'conversational': {
            'type': 'conversational',
            'fallback_prompt': 'shared/conversational/base.txt',
            'model': 'full',
            'temperature': 0.8
        }
    }
    
    for feature in features:
        if feature in generation_templates:
            config[feature] = generation_templates[feature]
    
    return config


def _create_prompt_folder(brand_id: str):
    """브랜드 프롬프트 폴더 생성"""
    prompt_dir = Path(f"prompts/{brand_id}")
    prompt_dir.mkdir(parents=True, exist_ok=True)
    
    # README 생성
    readme = prompt_dir / "README.md"
    readme.write_text(f"""# {brand_id.upper()} Custom Prompts

브랜드 전용 프롬프트를 이곳에 추가하세요.

## 파일 명명 규칙

- `factual_[subtype].txt` - 팩트 기반 프롬프트
- `insight_[subtype].txt` - 인사이트 프롬프트
- `conversational_[subtype].txt` - 대화형 프롬프트

## 예시

```
prompts/{brand_id}/
├── factual_product.txt
├── insight_advisor.txt
└── conversational_base.txt
```

## 설정 연결

configs/brands/{brand_id}.yaml에서 연결:

```yaml
generation:
  product_recommendation:
    prompt: {brand_id}/factual_product.txt
    fallback_prompt: shared/factual/product.txt
```
""")
    
    console.print(f"[green]✅ 프롬프트 폴더 생성: {prompt_dir}[/green]")


if __name__ == "__main__":
    app()
