### **scripts/test_brand.py**

#!/usr/bin/env python3
"""
Brand Test Script
브랜드 설정 테스트
"""

import typer
from rich.console import Console
from rich.panel import Panel
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.engine import UniversalEngine

app = typer.Typer()
console = Console()


@app.command()
def test(
    brand_id: str,
    question: str = "안녕하세요"
):
    """
    브랜드 엔진 테스트
    
    Args:
        brand_id: 브랜드 ID
        question: 테스트 질문
    """
    console.print(f"\n🧪 [bold cyan]Testing brand: {brand_id}[/bold cyan]\n")
    
    try:
        # 엔진 초기화
        console.print("Initializing engine...")
        engine = UniversalEngine.get_instance(brand_id)
        
        console.print(f"✅ Engine initialized: {engine.brand_id}")
        
        # 질문 실행
        console.print(f"\n[bold]Question:[/bold] {question}\n")
        
        response = engine.ask(question)
        
        # 결과 출력
        console.print(Panel(
            response.message,
            title=f"Response ({response.question_type})",
            border_style="green"
        ))
        
        # 메타데이터
        console.print(f"\n[dim]Question Type: {response.question_type}[/dim]")
        console.print(f"[dim]Processing Time: {response.processing_time:.3f}s[/dim]")
        
        if response.metadata:
            console.print(f"\n[bold]Metadata:[/bold]")
            for key, value in response.metadata.items():
                console.print(f"  {key}: {value}")
        
    except Exception as e:
        console.print(f"\n[red]❌ Test failed: {e}[/red]")
        raise


if __name__ == "__main__":
    app()
