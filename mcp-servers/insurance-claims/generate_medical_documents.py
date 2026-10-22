"""Phase 5B: generate fictional searchable PDFs and link them to SQLite.
Run from anywhere: python generate_medical_documents.py
Place this script alongside init_database.py in insurance-claims project.
"""
import json
import sqlite3
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parent
DB = ROOT / 'data' / 'insurance_claims.db'
DOCS = ROOT / 'data' / 'medical_documents'
MANIFEST = ROOT / 'data' / 'medical_document_manifest.json'

# Every person, clinic, date, and clinical fact below is fictional.
SCENARIOS = [
    dict(claim='CLM001', patient='Patient A (fictional)', outcome='prior_referral_present',
         docs=[
          ('consultation', '2026-01-05', 'Orthopaedic consultation', 'Dr. A. Malik, Orthopaedic Specialist',
           'Assessment: persistent right knee pain after a minor twisting injury. Examination suggests a soft-tissue strain. Plan: supervised physiotherapy.'),
          ('referral', '2026-01-05', 'Referral to physiotherapy', 'Dr. A. Malik, Orthopaedic Specialist',
           'I refer Patient A for six sessions of physiotherapy for right knee rehabilitation. Referral issued before treatment begins.'),
          ('treatment', '2026-01-09', 'Physiotherapy treatment record', 'R. Shah, Physiotherapist',
           'Initial physiotherapy session completed for right knee rehabilitation. Referral dated 2026-01-05 reviewed. Six sessions planned.'),
          ('claim_form', '2026-02-03', 'Insurance claim submission', 'Claims Intake Desk',
           'Reimbursement requested for physiotherapy sessions relating to right knee rehabilitation. Supporting referral and treatment notes attached.'),
         ]),
    dict(claim='CLM002', patient='Patient B (fictional)', outcome='prior_referral_present',
         docs=[
          ('consultation', '2026-02-02', 'Orthopaedic consultation', 'Dr. N. Qureshi, Orthopaedic Specialist',
           'Assessment: left shoulder stiffness and reduced range of motion. Conservative rehabilitation recommended.'),
          ('referral', '2026-02-02', 'Referral to physiotherapy', 'Dr. N. Qureshi, Orthopaedic Specialist',
           'Patient B is referred for four physiotherapy sessions for left shoulder mobility exercises.'),
          ('treatment', '2026-02-06', 'Physiotherapy discharge summary', 'S. Raza, Physiotherapist',
           'Four physiotherapy sessions completed following referral dated 2026-02-02. Functional improvement documented.'),
          ('claim_form', '2026-02-20', 'Insurance claim submission', 'Claims Intake Desk',
           'Claim submitted for completed physiotherapy treatment. Claim administrative status: Closed.'),
         ]),
    dict(claim='CLM003', patient='Patient C (fictional)', outcome='no_prior_referral_in_dataset',
         docs=[
          ('consultation', '2026-03-18', 'General practice consultation', 'Dr. K. Ahmed, General Practitioner',
           'Assessment: intermittent lower back discomfort. Advice: activity modification and follow-up if symptoms persist. No specialist physiotherapy referral issued in this note.'),
          ('treatment', '2026-03-10', 'Physiotherapy treatment record', 'M. Hussain, Physiotherapist',
           'Initial physiotherapy session for lower back discomfort on 2026-03-10. No orthopaedic referral was available in the records provided at intake.'),
          ('claim_form', '2026-03-25', 'Insurance claim submission', 'Claims Intake Desk',
           'Claim submitted for physiotherapy sessions starting 2026-03-10. Orthopaedic referral documentation not included in the submitted packet.'),
         ]),
    dict(claim='CLM004', patient='Patient D (fictional)', outcome='requires_manual_review',
         docs=[
          ('consultation', '2026-04-01', 'Orthopaedic consultation', 'Dr. S. Farooq, Orthopaedic Specialist',
           'Assessment: left ankle discomfort following exercise. Plan: consider rehabilitation if symptoms continue.'),
          ('referral', '2026-04-03', 'Referral to physiotherapy', 'Dr. S. Farooq, Orthopaedic Specialist',
           'Referral for physiotherapy of the LEFT ankle, up to five sessions, issued 2026-04-03.'),
          ('treatment', '2026-04-06', 'Physiotherapy treatment record', 'H. Khan, Physiotherapist',
           'Physiotherapy commenced for RIGHT ankle pain on 2026-04-06. Referral on file mentions LEFT ankle. Laterality discrepancy requires clarification.'),
          ('claim_form', '2026-04-19', 'Insurance claim submission', 'Claims Intake Desk',
           'Claim submitted for ankle physiotherapy. Submitted referral and treatment record describe different sides.'),
         ]),
]

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='DocTitleCustom', parent=styles['Heading1'], alignment=TA_CENTER, fontSize=15, leading=19, spaceAfter=15))
styles.add(ParagraphStyle(name='DocBodyCustom', parent=styles['BodyText'], fontSize=10.5, leading=16, spaceAfter=10))
styles.add(ParagraphStyle(name='DocFooterCustom', parent=styles['BodyText'], fontSize=8, textColor=colors.grey, leading=11))


def create_pdf(path, record, kind, date, title, author, content):
    doc = SimpleDocTemplate(str(path), pagesize=A4, rightMargin=22*mm, leftMargin=22*mm, topMargin=23*mm, bottomMargin=22*mm)
    fields = [
        ('Claim ID', record['claim']), ('Patient', record['patient']),
        ('Document type', kind.replace('_', ' ').title()), ('Document date', date),
        ('Issued by', author),
    ]
    table = Table([[Paragraph(f'<b>{k}</b>', styles['BodyText']), Paragraph(v, styles['BodyText'])] for k,v in fields], colWidths=[37*mm, 120*mm], hAlign='LEFT')
    table.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'), ('BOTTOMPADDING',(0,0),(-1,-1),8), ('LINEBELOW',(0,-1),(-1,-1),0.5,colors.lightgrey)]))
    story = [Paragraph('SYNTHETIC MEDICAL DOCUMENT', styles['DocTitleCustom']),
             Paragraph(title, styles['Heading2']), Spacer(1,4*mm), table,
             Spacer(1,10*mm), Paragraph('Clinical / administrative narrative', styles['Heading3']),
             Paragraph(content, styles['DocBodyCustom']), Spacer(1,15*mm),
             Paragraph('FICTIONAL TEST DATA - NOT A REAL MEDICAL RECORD, REFERRAL, OR INSURANCE DETERMINATION.', styles['DocFooterCustom'])]
    doc.build(story)


def main():
    if not DB.is_file():
        raise SystemExit(f'Missing database: {DB}\nRun python init_database.py first.')
    DOCS.mkdir(parents=True, exist_ok=True)
    manifest = []
    with sqlite3.connect(DB) as conn:
        conn.execute('PRAGMA foreign_keys = ON')
        conn.execute('''CREATE TABLE IF NOT EXISTS claim_documents (
            document_id TEXT PRIMARY KEY,
            claim_id TEXT NOT NULL REFERENCES claims(claim_id),
            document_type TEXT NOT NULL,
            document_date TEXT NOT NULL,
            title TEXT NOT NULL,
            file_path TEXT NOT NULL
        )''')
        known = {r[0] for r in conn.execute('SELECT claim_id FROM claims')}
        for record in SCENARIOS:
            if record['claim'] not in known:
                raise RuntimeError(f"Claim {record['claim']} missing from SQLite; initialize original dataset first")
            for idx, (kind, date, title, author, narrative) in enumerate(record['docs'], 1):
                doc_id = f"{record['claim']}-DOC{idx:02d}"
                path = DOCS / f'{doc_id}_{kind}.pdf'
                create_pdf(path, record, kind, date, title, author, narrative)
                relative_path = path.relative_to(ROOT).as_posix()
                conn.execute('''INSERT INTO claim_documents
                    (document_id, claim_id, document_type, document_date, title, file_path)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(document_id) DO UPDATE SET
                    claim_id=excluded.claim_id, document_type=excluded.document_type,
                    document_date=excluded.document_date, title=excluded.title,
                    file_path=excluded.file_path''',
                    (doc_id, record['claim'], kind, date, title, relative_path))
                manifest.append({'document_id':doc_id,'claim_id':record['claim'],'patient':record['patient'],
                    'document_type':kind,'document_date':date,'title':title,'file_path':relative_path,
                    'expected_scenario':record['outcome']})
        conn.commit()
        n = conn.execute('SELECT COUNT(*) FROM claim_documents').fetchone()[0]
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Generated {len(manifest)} searchable PDFs')
    print(f'SQLite document records: {n}')
    print(f'PDF directory: {DOCS}')
    print(f'Manifest: {MANIFEST}')
    for record in SCENARIOS:
        print(f"{record['claim']}: {len(record['docs'])} documents; expected scenario: {record['outcome']}")

if __name__ == '__main__':
    main()
