# 📝 scripts/init_db.py - Neo4j 초기화 스크립트

#!/usr/bin/env python3
"""
Neo4j Database Initialization Script
Neo4j 데이터베이스 초기 설정 및 벡터 인덱스 생성
"""

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
import sys
from pathlib import Path
import os

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.shared.neo4j import get_neo4j_client
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

app = typer.Typer()
console = Console()


@app.command()
def init(
    drop_existing: bool = typer.Option(
        False, 
        "--drop", 
        help="기존 인덱스/제약조건 삭제 후 재생성"
    ),
    vector_dimensions: int = typer.Option(
        1536, 
        "--dimensions", 
        help="벡터 임베딩 차원수"
    ),
    create_sample_data: bool = typer.Option(
        False,
        "--sample-data",
        help="샘플 데이터 생성"
    )
):
    """
    Neo4j 데이터베이스 초기화
    
    - 벡터 인덱스 생성
    - 필수 인덱스 생성
    - 제약조건 설정
    """
    console.print("\n🚀 [bold cyan]ONTIX Universal - Neo4j Initialization[/bold cyan]\n")
    
    try:
        # Neo4j 연결
        console.print("📡 Connecting to Neo4j...")
        neo4j = get_neo4j_client()
        
        # Health check
        health = neo4j.health_check()
        
        if health['status'] != 'healthy':
            console.print(f"[red]❌ Neo4j unhealthy: {health}[/red]")
            raise typer.Exit(1)
        
        console.print(f"[green]✅ Connected: {health['database']}[/green]\n")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            # 1. 기존 인덱스/제약조건 삭제 (옵션)
            if drop_existing:
                task = progress.add_task("Dropping existing indexes...", total=None)
                _drop_existing_indexes(neo4j)
                progress.update(task, completed=True)
                console.print("[yellow]⚠️  Dropped existing indexes[/yellow]")
            
            # 2. 벡터 인덱스 생성
            task = progress.add_task("Creating vector index...", total=None)
            _create_vector_index(neo4j, vector_dimensions)
            progress.update(task, completed=True)
            console.print("[green]✅ Vector index created[/green]")
            
            # 3. 노드 인덱스 생성
            task = progress.add_task("Creating node indexes...", total=None)
            _create_node_indexes(neo4j)
            progress.update(task, completed=True)
            console.print("[green]✅ Node indexes created[/green]")
            
            # 4. 제약조건 생성
            task = progress.add_task("Creating constraints...", total=None)
            _create_constraints(neo4j)
            progress.update(task, completed=True)
            console.print("[green]✅ Constraints created[/green]")
            
            # 5. 샘플 데이터 생성 (옵션)
            if create_sample_data:
                task = progress.add_task("Creating sample data...", total=None)
                _create_sample_data(neo4j)
                progress.update(task, completed=True)
                console.print("[green]✅ Sample data created[/green]")
        
        # 완료 메시지
        console.print("\n" + "="*60)
        console.print(Panel(
            "[bold green]✅ Neo4j Initialization Complete![/bold green]\n\n"
            "Next steps:\n"
            "1. python scripts/sync_brand.py <brand_id>\n"
            "2. python scripts/test_brand.py <brand_id> '안녕'\n"
            "3. python app/main.py",
            title="Success",
            border_style="green"
        ))
        console.print("="*60 + "\n")
        
    except Exception as e:
        console.print(f"\n[red]❌ Initialization failed: {e}[/red]")
        raise typer.Exit(1)


def _drop_existing_indexes(neo4j):
    """기존 인덱스 및 제약조건 삭제"""
    
    # 벡터 인덱스 삭제
    vector_index_name = os.getenv('NEO4J_VECTOR_INDEX', 'ontix_global_concept_index')
    
    try:
        neo4j.query(f"DROP INDEX {vector_index_name} IF EXISTS")
        console.print(f"  Dropped vector index: {vector_index_name}")
    except Exception as e:
        console.print(f"  [dim]Vector index not found: {e}[/dim]")
    
    # 일반 인덱스 목록 조회 및 삭제
    try:
        indexes = neo4j.query("SHOW INDEXES")
        for idx in indexes:
            idx_name = idx.get('name')
            if idx_name and not idx_name.startswith('constraint_'):
                try:
                    neo4j.query(f"DROP INDEX {idx_name} IF EXISTS")
                    console.print(f"  Dropped index: {idx_name}")
                except:
                    pass
    except Exception as e:
        console.print(f"  [dim]Could not list indexes: {e}[/dim]")
    
    # 제약조건 삭제
    try:
        constraints = neo4j.query("SHOW CONSTRAINTS")
        for const in constraints:
            const_name = const.get('name')
            if const_name:
                try:
                    neo4j.query(f"DROP CONSTRAINT {const_name} IF EXISTS")
                    console.print(f"  Dropped constraint: {const_name}")
                except:
                    pass
    except Exception as e:
        console.print(f"  [dim]Could not list constraints: {e}[/dim]")


def _create_vector_index(neo4j, dimensions: int):
    """벡터 인덱스 생성"""
    
    vector_index_name = os.getenv('NEO4J_VECTOR_INDEX', 'ontix_global_concept_index')
    
    # 벡터 인덱스 생성 쿼리
    query = f"""
    CREATE VECTOR INDEX {vector_index_name} IF NOT EXISTS
    FOR (c:Concept)
    ON c.embedding
    OPTIONS {{
        indexConfig: {{
            `vector.dimensions`: {dimensions},
            `vector.similarity_function`: 'cosine'
        }}
    }}
    """
    
    try:
        neo4j.query(query)
        console.print(f"  Vector index: {vector_index_name} ({dimensions}D)")
    except Exception as e:
        # Neo4j 5.x 이전 버전용 대체 방법
        console.print(f"  [yellow]Warning: {e}[/yellow]")
        console.print("  [dim]Trying alternative vector index creation...[/dim]")
        
        # 대체 방법 (Neo4j 5.11+)
        alt_query = f"""
        CALL db.index.vector.createNodeIndex(
            '{vector_index_name}',
            'Concept',
            'embedding',
            {dimensions},
            'cosine'
        )
        """
        try:
            neo4j.query(alt_query)
            console.print(f"  Vector index created (alternative method)")
        except Exception as e2:
            console.print(f"  [red]Failed to create vector index: {e2}[/red]")
            raise


def _create_node_indexes(neo4j):
    """노드 인덱스 생성"""
    
    indexes = [
        # Brand
        "CREATE INDEX brand_id_index IF NOT EXISTS FOR (b:Brand) ON (b.id)",
        
        # Post
        "CREATE INDEX post_brand_id_index IF NOT EXISTS FOR (p:Post) ON (p.brand_id)",
        "CREATE INDEX post_likes_index IF NOT EXISTS FOR (p:Post) ON (p.likes)",
        "CREATE INDEX post_created_at_index IF NOT EXISTS FOR (p:Post) ON (p.created_at)",
        
        # Concept
        "CREATE INDEX concept_brand_id_index IF NOT EXISTS FOR (c:Concept) ON (c.brand_id)",
        "CREATE INDEX concept_category_index IF NOT EXISTS FOR (c:Concept) ON (c.category)",
        
        # Product
        "CREATE INDEX product_brand_id_index IF NOT EXISTS FOR (p:Product) ON (p.brand_id)",
        "CREATE INDEX product_stock_index IF NOT EXISTS FOR (p:Product) ON (p.stock)",
        "CREATE INDEX product_price_index IF NOT EXISTS FOR (p:Product) ON (p.price)",
        "CREATE INDEX product_category_index IF NOT EXISTS FOR (p:Product) ON (p.category)",
    ]
    
    for idx_query in indexes:
        try:
            neo4j.query(idx_query)
            # 인덱스 이름 추출
            idx_name = idx_query.split()[2]
            console.print(f"  Index: {idx_name}")
        except Exception as e:
            console.print(f"  [yellow]Warning: {e}[/yellow]")


def _create_constraints(neo4j):
    """제약조건 생성 (UNIQUE)"""
    
    constraints = [
        # Brand
        "CREATE CONSTRAINT brand_id_unique IF NOT EXISTS FOR (b:Brand) REQUIRE b.id IS UNIQUE",
        
        # Post
        "CREATE CONSTRAINT post_id_unique IF NOT EXISTS FOR (p:Post) REQUIRE p.id IS UNIQUE",
        
        # Concept
        "CREATE CONSTRAINT concept_id_unique IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE",
        
        # Product
        "CREATE CONSTRAINT product_id_unique IF NOT EXISTS FOR (p:Product) REQUIRE p.id IS UNIQUE",
    ]
    
    for const_query in constraints:
        try:
            neo4j.query(const_query)
            # 제약조건 이름 추출
            const_name = const_query.split()[2]
            console.print(f"  Constraint: {const_name}")
        except Exception as e:
            console.print(f"  [yellow]Warning: {e}[/yellow]")


def _create_sample_data(neo4j):
    """샘플 데이터 생성"""
    
    # Brand 노드
    neo4j.query("""
        MERGE (b:Brand {id: 'sample'})
        SET b.name = 'Sample Brand',
            b.description = 'A sample brand for testing',
            b.industry = 'Technology',
            b.created_at = datetime()
    """)
    console.print("  Created: Sample Brand")
    
    # Concept 노드 (벡터 없이)
    neo4j.query("""
        MERGE (c:Concept {id: 'sample_concept_1'})
        SET c.brand_id = 'sample',
            c.text = 'This is a sample concept for testing',
            c.category = 'general',
            c.created_at = datetime()
    """)
    console.print("  Created: Sample Concept")
    
    # Post 노드
    neo4j.query("""
        MERGE (p:Post {id: 'sample_post_1'})
        SET p.brand_id = 'sample',
            p.content = 'This is a sample post',
            p.likes = 100,
            p.comments = 10,
            p.created_at = datetime()
    """)
    console.print("  Created: Sample Post")
    
    # Product 노드
    neo4j.query("""
        MERGE (pr:Product {id: 'sample_product_1'})
        SET pr.brand_id = 'sample',
            pr.name = 'Sample Product',
            pr.price = 99000,
            pr.stock = 50,
            pr.category = 'electronics',
            pr.created_at = datetime()
    """)
    console.print("  Created: Sample Product")


@app.command()
def check():
    """
    Neo4j 상태 확인
    """
    console.print("\n🔍 [bold cyan]Checking Neo4j Status[/bold cyan]\n")
    
    try:
        neo4j = get_neo4j_client()
        
        # Health check
        health = neo4j.health_check()
        console.print(f"Status: [green]{health['status']}[/green]")
        console.print(f"Database: {health['database']}")
        
        # 벡터 인덱스 확인
        vector_index_name = os.getenv('NEO4J_VECTOR_INDEX', 'ontix_global_concept_index')
        
        console.print(f"\n📊 Indexes:")
        indexes = neo4j.query("SHOW INDEXES")
        
        vector_exists = False
        for idx in indexes:
            idx_name = idx.get('name', 'Unknown')
            idx_type = idx.get('type', 'Unknown')
            idx_state = idx.get('state', 'Unknown')
            
            if idx_name == vector_index_name:
                vector_exists = True
                console.print(f"  ✅ {idx_name} ({idx_type}) - {idx_state}")
            else:
                console.print(f"  • {idx_name} ({idx_type}) - {idx_state}")
        
        if not vector_exists:
            console.print(f"  [red]❌ Vector index not found: {vector_index_name}[/red]")
        
        # 노드 수 확인
        console.print(f"\n📈 Node Counts:")
        
        counts = {
            'Brand': neo4j.query("MATCH (b:Brand) RETURN count(b) as count")[0]['count'],
            'Post': neo4j.query("MATCH (p:Post) RETURN count(p) as count")[0]['count'],
            'Concept': neo4j.query("MATCH (c:Concept) RETURN count(c) as count")[0]['count'],
            'Product': neo4j.query("MATCH (p:Product) RETURN count(p) as count")[0]['count'],
        }
        
        for label, count in counts.items():
            console.print(f"  {label}: {count}")
        
        console.print("\n✅ Check complete!\n")
        
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def reset():
    """
    ⚠️ 전체 데이터베이스 초기화 (위험)
    모든 노드와 관계를 삭제합니다.
    """
    console.print("\n[bold red]⚠️  WARNING: This will DELETE ALL DATA![/bold red]\n")
    
    confirm = typer.confirm("Are you sure you want to reset the database?")
    
    if not confirm:
        console.print("[yellow]Cancelled[/yellow]")
        return
    
    double_confirm = typer.confirm("Really? This cannot be undone!")
    
    if not double_confirm:
        console.print("[yellow]Cancelled[/yellow]")
        return
    
    try:
        neo4j = get_neo4j_client()
        
        console.print("\n🗑️  Deleting all nodes and relationships...")
        
        # 모든 노드 및 관계 삭제
        neo4j.query("MATCH (n) DETACH DELETE n")
        
        console.print("[green]✅ All data deleted[/green]")
        console.print("\nRun 'python scripts/init_db.py init' to reinitialize\n")
        
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
