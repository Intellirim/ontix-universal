### **scripts/sync_brand.py**

#!/usr/bin/env python3
"""
Brand Sync Script
브랜드 데이터 동기화
"""

import typer
from rich.console import Console
from rich.progress import Progress
import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.platform.config_manager import ConfigManager

app = typer.Typer()
console = Console()


@app.command()
def sync(
    brand_id: str,
    force: bool = False
):
    """
    브랜드 데이터 Neo4j에 동기화
    
    Args:
        brand_id: 브랜드 ID
        force: 강제 동기화 (기존 데이터 덮어쓰기)
    """
    console.print(f"\n🔄 [bold cyan]Syncing brand: {brand_id}[/bold cyan]\n")
    
    try:
        # Config 로드
        config = ConfigManager.load_config(brand_id)
        
        console.print(f"✅ Config loaded: {config['brand']['name']}")
        
        # Neo4j 연결
        from app.services.shared.neo4j import get_neo4j_client
        
        neo4j = get_neo4j_client()
        
        # Health check
        health = neo4j.health_check()
        
        if health['status'] != 'healthy':
            console.print(f"[red]❌ Neo4j unhealthy: {health}[/red]")
            return
        
        console.print(f"✅ Neo4j connected: {health['database']}")
        
        # 브랜드 노드 생성/업데이트
        with Progress() as progress:
            task = progress.add_task("[cyan]Syncing brand...", total=100)
            
            # Brand 노드
            query = """
            MERGE (b:Brand {id: $brand_id})
            SET b.name = $name,
                b.description = $description,
                b.industry = $industry,
                b.updated_at = datetime()
            RETURN b
            """
            
            neo4j.query(query, {
                'brand_id': brand_id,
                'name': config['brand']['name'],
                'description': config['brand'].get('description', ''),
                'industry': config['brand'].get('industry', '')
            })
            
            progress.update(task, advance=50)
            
            # 인덱스 생성
            indexes = [
                "CREATE INDEX IF NOT EXISTS FOR (b:Brand) ON (b.id)",
                f"CREATE INDEX IF NOT EXISTS FOR (p:Post) ON (p.brand_id)",
                f"CREATE INDEX IF NOT EXISTS FOR (c:Concept) ON (c.brand_id)",
                f"CREATE INDEX IF NOT EXISTS FOR (pr:Product) ON (pr.brand_id)"
            ]
            
            for idx_query in indexes:
                try:
                    neo4j.query(idx_query)
                except Exception as e:
                    console.print(f"[yellow]Warning: {e}[/yellow]")
            
            progress.update(task, advance=50)
        
        console.print(f"\n[green]✅ Sync completed: {brand_id}[/green]")
        
    except FileNotFoundError:
        console.print(f"[red]❌ Brand config not found: {brand_id}[/red]")
    except Exception as e:
        console.print(f"[red]❌ Sync failed: {e}[/red]")
        raise


if __name__ == "__main__":
    app()
