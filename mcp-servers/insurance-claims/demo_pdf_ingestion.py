"""Reproducible V2 demonstration with real Ollama, Chroma, OCR, MCP and LangGraph.
Only synthetic data in temporary storage is written; original data is compared.
"""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile

import chromadb
from langchain_core.messages import HumanMessage, SystemMessage
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from ollama import AsyncClient

from pdf_ingestion import ingest_pdf, ensure_schema
from test_pdf_ingestion import REFERRAL, TREATMENT, make_pdf
from langgraph_rag_agent import SYSTEM_PROMPT, create_agent, get_referral_validation_report

ROOT = Path(__file__).resolve().parent


def original_snapshot():
    data = ROOT / 'data'
    with sqlite3.connect(f'file:{data / "insurance_claims.db"}?mode=ro', uri=True) as conn:
        claims = conn.execute('SELECT * FROM claims ORDER BY claim_id').fetchall()
        documents = conn.execute('SELECT * FROM claim_documents ORDER BY document_id').fetchall()
    collection = chromadb.PersistentClient(path=str(data / 'chroma_db')).get_collection('insurance_medical_documents')
    records = collection.get(include=['documents', 'metadatas', 'embeddings'])
    vectors = {i: {'text': t, 'metadata': m, 'embedding': e.tolist()}
               for i, t, m, e in zip(records['ids'], records['documents'], records['metadatas'], records['embeddings'])}
    return {'claims': claims, 'documents': documents, 'vectors': vectors,
            'pdfs': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in data.rglob('*.pdf')}}


async def demonstrate():
    before = original_snapshot()
    with tempfile.TemporaryDirectory(prefix='insurance-v2-demo-') as temp:
        root = Path(temp)
        db = root / 'claims.db'
        chroma = root / 'chroma'
        with sqlite3.connect(db) as conn:
            conn.execute('CREATE TABLE claims(claim_id TEXT PRIMARY KEY, status TEXT, amount INTEGER, currency TEXT)')
            conn.executemany('INSERT INTO claims VALUES(?,?,?,?)',
                             [('CLM101', 'Open', 10, 'USD'), ('CLM102', 'Open', 20, 'USD')])
            ensure_schema(conn)
        pdf = make_pdf(root / 'unseen-referral-packet.pdf', [REFERRAL, TREATMENT])
        result = ingest_pdf(pdf, 'CLM101', db_path=db, chroma_path=chroma)
        assert result['status'] == 'complete' and result['chunk_count'] == 2
        assert ingest_pdf(pdf, 'CLM101', db_path=db, chroma_path=chroma)['status'] == 'duplicate'
        with sqlite3.connect(db) as conn:
            assert conn.execute('SELECT count(*) FROM claim_documents').fetchone()[0] == 1
        collection = chromadb.PersistentClient(path=str(chroma)).get_collection('insurance_medical_documents')
        assert collection.count() == 2
        print('PASS: real Ollama embeddings, SQLite registration, Chroma indexing and duplicate detection')
        env = {**os.environ, 'INSURANCE_DB_PATH': str(db), 'INSURANCE_CHROMA_PATH': str(chroma)}
        params = StdioServerParameters(command=sys.executable, args=[str(ROOT / 'rag_server.py')], env=env)
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                async def call(name, **arguments):
                    response = await session.call_tool(name, arguments=arguments)
                    assert not response.is_error, name
                    return json.loads('\n'.join(c.text for c in response.content if c.type == 'text'))
                evidence = await call('get_claim_medical_evidence', claim_id='CLM101')
                assert {r['page'] for r in evidence['results']} == {1, 2}
                assert not (await call('get_claim_medical_evidence', claim_id='CLM102'))['results']
                hits = await call('search_medical_documents', claim_id='CLM101', question='specialist referral')
                assert hits['results'] and all(r['claim_id'] == 'CLM101' for r in hits['results'])
                assert not (await call('search_medical_documents', claim_id='CLM102', question='specialist referral'))['results']
                finding = await call('validate_claim_referral', claim_id='CLM101')
                assert finding['overall_finding'] == 'PASS'
                other = await call('validate_claim_referral', claim_id='CLM102')
                assert other['overall_finding'] == 'REVIEW' and not other['citations']
                print('PASS: MCP evidence retrieval, semantic search, claim isolation and deterministic validation')
                tools = await session.list_tools()
                graph = create_agent(session, AsyncClient(), tools.tools)
                state = await graph.ainvoke({'messages': [SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content='Validate the specialist referral for CLM101.')]}, config={'recursion_limit': 13})
                report = get_referral_validation_report(state['messages'])
                assert 'Overall finding: PASS' in report
                assert 'unseen-referral-packet.pdf, page 1:' in report
                assert 'page 2:' in report
                print('PASS: LangGraph workflow and genuine document/page citations')
                scanned = make_pdf(root / 'scanned-referral.pdf', [REFERRAL], scanned={1})
                scan_result = ingest_pdf(scanned, 'CLM102', db_path=db, chroma_path=chroma)
                assert scan_result['ocr_pages'] == [1]
                scan_evidence = await call('get_claim_medical_evidence', claim_id='CLM102')
                assert 'RIGHT knee' in scan_evidence['results'][0]['text']
                assert scan_evidence['results'][0]['page'] == 1
                scan_finding = await call('validate_claim_referral', claim_id='CLM102')
                assert scan_finding['overall_finding'] == 'REVIEW'
                print('PASS: scanned PDF OCR, local embeddings, MCP retrieval and conservative REVIEW')
    assert original_snapshot() == before, 'Original demonstration data changed'
    print(f'PASS: original {len(before["documents"])} registrations/PDFs and {len(before["vectors"])} vectors unchanged')


if __name__ == '__main__':
    asyncio.run(demonstrate())
