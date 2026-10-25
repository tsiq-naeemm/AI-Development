"""Isolated V2 tests using synthetic PDFs, actual OCR/Chroma and fake embeddings."""

import json
from pathlib import Path
import shutil
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import chromadb
import pymupdf as fitz
import pdf_ingestion as ingestion
import rag_server
from langgraph_rag_agent import format_referral_validation

REFERRAL = (
    "Document type: Referral\nDocument date: 2026-01-05\n"
    "Issued by Dr. A. Malik, Orthopaedic Specialist\n"
    "I refer this fictional patient for RIGHT knee physiotherapy."
)
TREATMENT = (
    "Document type: Treatment\nDocument date: 2026-01-09\n"
    "Physiotherapy commenced for RIGHT knee rehabilitation."
)


def make_pdf(path, texts, scanned=()):
    with fitz.open() as pdf:
        for index, text in enumerate(texts, 1):
            page = pdf.new_page()
            if index in scanned:
                with fitz.open() as image_pdf:
                    image_page = image_pdf.new_page()
                    image_page.insert_text((40, 60), text, fontsize=13)
                    png = image_page.get_pixmap(dpi=180).tobytes("png")
                page.insert_image(page.rect, stream=png)
            else:
                page.insert_text((40, 60), text, fontsize=13)
        pdf.save(path)
    return path


class Embeddings:
    def embed(self, **kwargs):
        texts = kwargs["input"]
        if isinstance(texts, str):
            texts = [texts]
        return SimpleNamespace(embeddings=[[1.0, 0.1, 0.2] for _ in texts])


class TestPDFIngestion(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.db = self.root / "claims.db"
        self.chroma = self.root / "chroma"
        with sqlite3.connect(self.db) as conn:
            conn.execute("CREATE TABLE claims(claim_id TEXT PRIMARY KEY)")
            conn.executemany("INSERT INTO claims VALUES(?)", [("CLM101",), ("CLM102",)])
            ingestion.ensure_schema(conn)
        self.collection = chromadb.PersistentClient(
            path=str(self.chroma)
        ).get_or_create_collection(
            ingestion.COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
        self.patches = [
            patch.object(rag_server, "DB_PATH", self.db),
            patch.object(rag_server, "CHROMA_PATH", self.chroma),
            patch.object(rag_server, "Client", return_value=Embeddings()),
        ]
        for item in self.patches:
            item.start()
        self.pdf = make_pdf(self.root / "previously_unseen.pdf", [REFERRAL, TREATMENT])

    def tearDown(self):
        for item in self.patches:
            item.stop()
        self.tmp.cleanup()

    def ingest(self, path=None, claim="CLM101", **kwargs):
        return ingestion.ingest_pdf(
            path or self.pdf,
            claim,
            db_path=self.db,
            collection=kwargs.pop("collection", self.collection),
            embedder=kwargs.pop("embedder", Embeddings()),
            **kwargs,
        )

    def test_text_registration_index_and_citations(self):
        result = self.ingest()
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["page_count"], 2)
        with sqlite3.connect(self.db) as conn:
            self.assertEqual(
                conn.execute("SELECT count(*) FROM claim_documents").fetchone()[0], 1
            )
        self.assertEqual(self.collection.count(), 2)
        evidence = rag_server.get_claim_medical_evidence(" clm101 ")
        self.assertEqual([r["page"] for r in evidence["results"]], [1, 2])
        self.assertTrue(
            all(r["source"] == str(self.pdf.resolve()) for r in evidence["results"])
        )
        finding = rag_server.validate_claim_referral("CLM101")
        self.assertEqual(finding["overall_finding"], "PASS")
        self.assertEqual(len(finding["citations"]), 2)
        report = format_referral_validation(finding)
        self.assertIn("previously_unseen.pdf, page 1:", report)
        self.assertIn("page 2:", report)
        self.assertIn("Dr. A. Malik", report)

    def test_incremental_ingestion_and_legacy_builder_preserve_each_other(self):
        import build_vector_index as builder

        legacy = make_pdf(
            self.root / "legacy.pdf",
            ["Fictional consultation for RIGHT knee discomfort."],
        )
        with sqlite3.connect(self.db) as conn:
            conn.execute(
                "INSERT INTO claim_documents VALUES(?,?,?,?,?,?)",
                (
                    "CLM101-DOC01",
                    "CLM101",
                    "consultation",
                    "2026-01-01",
                    "Legacy consultation",
                    str(legacy),
                ),
            )
        metadata = {
            "document_id": "CLM101-DOC01",
            "claim_id": "CLM101",
            "document_type": "consultation",
            "document_date": "2026-01-01",
            "title": "Legacy consultation",
            "file_path": str(legacy),
            "page": 1,
        }
        self.collection.upsert(
            ids=["CLM101-DOC01-P1-C1"],
            documents=["Fictional consultation"],
            metadatas=[metadata],
            embeddings=[[1.0, 0.1, 0.2]],
        )
        dynamic = self.ingest()
        self.assertEqual(self.collection.count(), 3)
        self.assertEqual(
            self.collection.get(ids=["CLM101-DOC01-P1-C1"])["documents"],
            ["Fictional consultation"],
        )
        before = self.collection.get(
            where={"document_id": dynamic["document_id"]},
            include=["documents", "metadatas"],
        )
        with (
            patch.object(builder, "DB_PATH", self.db),
            patch.object(builder, "CHROMA_PATH", self.chroma),
            patch.object(builder, "Client", return_value=Embeddings()),
        ):
            builder.main()
        self.assertEqual(self.collection.count(), 3)
        after = self.collection.get(
            where={"document_id": dynamic["document_id"]},
            include=["documents", "metadatas"],
        )
        self.assertEqual(before, after)

    def test_duplicate_and_renamed_duplicate(self):
        first = self.ingest()
        renamed = self.root / "renamed.pdf"
        shutil.copyfile(self.pdf, renamed)
        second = self.ingest(renamed)
        self.assertEqual(second["status"], "duplicate")
        self.assertEqual(first["document_id"], second["document_id"])
        self.assertEqual(self.collection.count(), 2)

    def test_scanned_actual_ocr(self):
        scanned = make_pdf(self.root / "scan.pdf", [REFERRAL], scanned={1})
        with fitz.open(scanned) as pdf:
            self.assertFalse(pdf[0].get_text().strip())
        result = self.ingest(scanned)
        self.assertEqual(result["ocr_pages"], [1])
        text = rag_server.get_claim_medical_evidence("CLM101")["results"][0]["text"]
        self.assertIn("RIGHT knee", text)
        self.assertEqual(
            rag_server.validate_claim_referral("CLM101")["overall_finding"], "REVIEW"
        )

    def test_mixed_pages(self):
        mixed = make_pdf(self.root / "mixed.pdf", [REFERRAL, TREATMENT], scanned={2})
        result = self.ingest(mixed)
        self.assertEqual(result["ocr_pages"], [2])
        records = rag_server.get_claim_medical_evidence("CLM101")["results"]
        self.assertEqual({r["page"] for r in records}, {1, 2})
        self.assertEqual({r["extraction_method"] for r in records}, {"text", "ocr"})
        self.assertEqual(
            rag_server.validate_claim_referral("CLM101")["overall_finding"], "REVIEW"
        )

    def test_claim_isolation_all_endpoints(self):
        self.ingest()
        self.assertEqual(rag_server.get_claim_medical_evidence("CLM102")["results"], [])
        self.assertEqual(
            rag_server.search_medical_documents("CLM102", "referral")["results"], []
        )
        result = rag_server.validate_claim_referral("CLM102")
        self.assertEqual(result["overall_finding"], "REVIEW")
        self.assertEqual(result["citations"], [])
        self.assertNotIn("A. Malik", json.dumps(result))
        matches = rag_server.search_medical_documents("CLM101", "referral")["results"]
        self.assertTrue(matches)
        self.assertTrue(all(r["claim_id"] == "CLM101" for r in matches))

    def test_isolation_with_both_claims_indexed(self):
        self.ingest()
        other = make_pdf(
            self.root / "claim-b.pdf", [TREATMENT.replace("RIGHT knee", "LEFT ankle")]
        )
        self.ingest(other, claim="CLM102")
        a = rag_server.validate_claim_referral("CLM101")
        b = rag_server.validate_claim_referral("CLM102")
        self.assertEqual(a["overall_finding"], "PASS")
        self.assertEqual(b["overall_finding"], "REVIEW")
        self.assertEqual(b["checks"]["referral_presence"]["finding"], "REVIEW")
        for endpoint in (
            lambda: rag_server.get_claim_medical_evidence("CLM102"),
            lambda: rag_server.search_medical_documents(
                "CLM102", "RIGHT knee specialist referral"
            ),
        ):
            result = endpoint()
            self.assertTrue(result["results"])
            self.assertTrue(all(r["claim_id"] == "CLM102" for r in result["results"]))
            self.assertNotIn("A. Malik", json.dumps(result))
            self.assertNotIn("RIGHT knee", json.dumps(result))

    def test_same_pdf_explicitly_attached_to_two_claims(self):
        a = self.ingest()
        b = self.ingest(claim="CLM102")
        self.assertNotEqual(a["document_id"], b["document_id"])
        self.assertEqual(self.collection.count(), 4)
        self.assertTrue(
            all(
                r["document_id"] == b["document_id"]
                for r in rag_server.get_claim_medical_evidence("CLM102")["results"]
            )
        )

    def test_invalid_unknown_claims(self):
        for claim in ("ABC123", "", "CLM999", None):
            with self.subTest(claim=claim), self.assertRaises(ValueError):
                self.ingest(claim=claim)
        self.assertEqual(self.collection.count(), 0)
        for endpoint in (
            rag_server.get_claim_medical_evidence,
            rag_server.validate_claim_referral,
        ):
            with self.assertRaises(ValueError):
                endpoint("ABC123")
        with self.assertRaises(ValueError):
            rag_server.get_claim_medical_evidence("CLM999")
        self.assertIn(
            "error", rag_server.search_medical_documents("CLM999", "referral")
        )

    def test_missing_file(self):
        with self.assertRaisesRegex(ingestion.IngestionError, "Cannot read PDF"):
            self.ingest(self.root / "missing.pdf")
        self.assertEqual(self.collection.count(), 0)

    def test_corrupted_file(self):
        path = self.root / "bad.pdf"
        path.write_bytes(b"%PDF-1.7 corrupted")
        with self.assertRaisesRegex(ingestion.IngestionError, "Cannot open PDF"):
            self.ingest(path)
        self.assertEqual(self.collection.count(), 0)
        self.assertTrue(
            rag_server.get_claim_medical_evidence("CLM101")["incomplete_ingestions"]
        )

    def test_encrypted_pdf(self):
        path = self.root / "encrypted.pdf"
        with fitz.open(self.pdf) as pdf:
            pdf.save(
                path,
                encryption=fitz.PDF_ENCRYPT_AES_256,
                owner_pw="owner",
                user_pw="secret",
            )
        with self.assertRaisesRegex(ingestion.IngestionError, "Encrypted PDF"):
            self.ingest(path)

    def test_blank_page_aborts_whole_document(self):
        path = make_pdf(self.root / "blank.pdf", [REFERRAL, ""])
        with self.assertRaisesRegex(ingestion.IngestionError, "Page 2"):
            self.ingest(path)
        self.assertEqual(self.collection.count(), 0)
        self.assertEqual(rag_server.get_claim_medical_evidence("CLM101")["results"], [])

    def test_ocr_unavailable_explicit(self):
        path = make_pdf(self.root / "scan.pdf", [REFERRAL], scanned={1})
        with patch.object(ingestion.shutil, "which", return_value=None):
            with self.assertRaisesRegex(
                ingestion.IngestionError, "Page 1: Local OCR unavailable"
            ):
                self.ingest(path)

    def test_partial_vector_failure_hidden_and_retry(self):
        actual = self.collection

        class FailingCollection:
            def upsert(self, **kwargs):
                actual.upsert(**kwargs)
                raise RuntimeError("simulated failure after vector write")

        with self.assertRaises(ingestion.IngestionError):
            self.ingest(collection=FailingCollection())
        self.assertEqual(actual.count(), 2)
        self.assertEqual(rag_server.get_claim_medical_evidence("CLM101")["results"], [])
        self.assertEqual(
            rag_server.search_medical_documents("CLM101", "referral")["results"], []
        )
        self.assertEqual(
            rag_server.validate_claim_referral("CLM101")["overall_finding"], "REVIEW"
        )
        self.assertEqual(self.ingest()["status"], "complete")
        self.assertEqual(actual.count(), 2)
        self.assertEqual(
            rag_server.validate_claim_referral("CLM101")["overall_finding"], "PASS"
        )

    def test_failed_additional_document_blocks_pass(self):
        self.ingest()
        path = make_pdf(self.root / "unreadable.pdf", [""])
        with self.assertRaises(ingestion.IngestionError):
            self.ingest(path)
        finding = rag_server.validate_claim_referral("CLM101")
        self.assertEqual(finding["overall_finding"], "REVIEW")
        self.assertIn("evidence_quality", finding["checks"])

    def test_missing_or_conflicting_date_review(self):
        for i, replacement in enumerate(
            ("Not a known date", "2026-02-31", "2026-01-05\nDocument date: 2026-01-06")
        ):
            path = make_pdf(
                self.root / f"dates{i}.pdf",
                [REFERRAL.replace("2026-01-05", replacement), TREATMENT],
            )
            self.ingest(path)
            self.assertEqual(
                rag_server.validate_claim_referral("CLM101")["overall_finding"],
                "REVIEW",
            )

    def test_laterality_mismatch_review(self):
        path = make_pdf(
            self.root / "mismatch.pdf", [REFERRAL.replace("RIGHT", "LEFT"), TREATMENT]
        )
        self.ingest(path)
        result = rag_server.validate_claim_referral("CLM101")
        self.assertEqual(result["overall_finding"], "REVIEW")
        self.assertEqual(result["checks"]["anatomy"]["referral_anatomy"], "LEFT knee")

    def test_unknown_structure_searchable_review(self):
        path = make_pdf(
            self.root / "unstructured.pdf",
            ["Notes from a fictional visit concerning RIGHT knee pain."],
        )
        self.ingest(path)
        self.assertTrue(
            rag_server.search_medical_documents("CLM101", "knee")["results"]
        )
        self.assertEqual(
            rag_server.validate_claim_referral("CLM101")["overall_finding"], "REVIEW"
        )

    def test_long_page_chunks_keep_page_number(self):
        path = make_pdf(
            self.root / "long.pdf", [("RIGHT knee rehabilitation. " * 3 + "\n") * 20]
        )
        self.ingest(path)
        records = rag_server.get_claim_medical_evidence("CLM101")["results"]
        self.assertGreater(len(records), 1)
        self.assertEqual({r["page"] for r in records}, {1})

    def test_standalone_search_requires_claim(self):
        import search_documents

        with self.assertRaises(ValueError):
            search_documents.search_documents("referral")

    def test_conflicting_specialists_review(self):
        text = REFERRAL + "\nIssued by Dr. B. Smith, Orthopaedic Specialist"
        path = make_pdf(self.root / "authors.pdf", [text, TREATMENT])
        self.ingest(path)
        finding = rag_server.validate_claim_referral("CLM101")
        self.assertEqual(finding["overall_finding"], "REVIEW")
        self.assertEqual(finding["checks"]["specialist"]["finding"], "REVIEW")

    def test_conflicting_treatment_anatomy_review(self):
        path = make_pdf(
            self.root / "sides.pdf",
            [REFERRAL, TREATMENT + "\nLEFT knee also documented."],
        )
        self.ingest(path)
        self.assertEqual(
            rag_server.validate_claim_referral("CLM101")["overall_finding"], "REVIEW"
        )

    def test_pending_vectors_do_not_leak_beside_complete_evidence(self):
        self.ingest()
        bad = make_pdf(
            self.root / "pending.pdf",
            ["unpublished medical text from another document"],
        )
        actual = self.collection

        class FailingCollection:
            def upsert(self, **kwargs):
                actual.upsert(**kwargs)
                raise RuntimeError("simulated interruption")

        with self.assertRaises(ingestion.IngestionError):
            self.ingest(bad, collection=FailingCollection())
        for endpoint in (
            lambda: rag_server.get_claim_medical_evidence("CLM101"),
            lambda: rag_server.search_medical_documents("CLM101", "medical text"),
        ):
            result = endpoint()
            self.assertEqual(len(result["results"]), 2)
            self.assertNotIn("unpublished", json.dumps(result["results"]))
        self.assertEqual(
            rag_server.validate_claim_referral("CLM101")["overall_finding"], "REVIEW"
        )

    def test_same_page_selectable_header_and_scanned_body(self):
        path = make_pdf(self.root / "hybrid.pdf", [REFERRAL], scanned={1})
        hybrid = self.root / "header-and-scan.pdf"
        with fitz.open(path) as pdf:
            pdf[0].insert_text(
                (40, 20), "Selectable header on scanned evidence", fontsize=10
            )
            pdf.save(hybrid)
        self.assertEqual(self.ingest(hybrid)["ocr_pages"], [1])
        evidence = rag_server.get_claim_medical_evidence("CLM101")
        self.assertIn("RIGHT knee", evidence["results"][0]["text"])

    def test_no_evidence_report(self):
        finding = rag_server.validate_claim_referral("CLM102")
        self.assertEqual(finding["overall_finding"], "REVIEW")
        self.assertIn("Overall finding: REVIEW", format_referral_validation(finding))


if __name__ == "__main__":
    unittest.main()
