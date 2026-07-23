import os
import sys

# Make the src layout importable without installation.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from savecontext.service import VaultService
from savecontext.storage import Store


@pytest.fixture
def service():
    svc = VaultService(store=Store(db_path=":memory:"))
    yield svc
    svc.close()


CONTRACT = """MASTER SERVICES AGREEMENT

This Agreement is entered into on January 15, 2024 between Acme Corporation
("Provider") and Globex Inc ("Customer").

1. Fees
The Customer shall pay $50,000 per month. Late payments accrue interest at 1.5% per month.
Payment is due within 30 days of invoice.

2. Liability
Provider's total liability shall not exceed $500,000 in aggregate. Provider shall
indemnify Customer against third-party claims. The Customer may not assign this
Agreement without written consent.

3. Term
This Agreement is effective from 2024-02-01 and continues for 24 months.
Either party may terminate with 90 days notice. Confidential information must not
be disclosed to any third party.

Contact: legal@acme.example.com or visit https://acme.example.com/legal
"""

CODE = """def compute_tax(amount, rate=0.2):
    # apply the standard rate
    total_tax = amount * rate
    return total_tax

class InvoiceProcessor:
    MAX_RETRIES = 3
    def process(self, invoice_id):
        return self.db.fetch(invoice_id)

See src/billing/processor.py and config.yaml for details.
"""


@pytest.fixture
def contract_text():
    return CONTRACT


@pytest.fixture
def code_text():
    return CODE
