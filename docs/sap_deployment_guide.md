# SAP Deployment Guide: FinanceBench-LLM

This guide describes how to deploy the FinanceBench-LLM financial QA system on SAP enterprise infrastructure, combining public SEC filing data with private ERP financial data for an **"Enterprise Financial Intelligence"** platform.

---

## Architecture Overview

```
                    +-----------------------+
                    |   SAP BTP / Kyma      |
                    |   (Hosting Platform)   |
                    +----------+------------+
                               |
              +----------------+----------------+
              |                                 |
    +---------v---------+           +-----------v-----------+
    | SAP AI Core       |           | Gradio Web App        |
    | (Model Serving)   |           | (User Interface)      |
    | - LoRA Adapter    |           | - Financial QA Tab    |
    | - NIM Runtime     |           | - Eval Results Tab    |
    +-------------------+           | - Comparison Tab      |
              |                     +-----------------------+
              |
    +---------v---------+
    | SAP HANA Cloud    |
    | Vector Engine     |
    | (RAG Pipeline)    |
    | - SEC Embeddings  |
    | - ERP Context     |
    +-------------------+
              |
    +---------v---------+
    | SAP S/4HANA       |
    | (Financial Data)  |
    | - P&L Statements  |
    | - Balance Sheets  |
    | - Journal Entries |
    +-------------------+
```

---

## 1. Deploy Model on SAP AI Core

SAP AI Core provides MLOps capabilities for training, deploying, and monitoring AI models.

### Prerequisites

- SAP BTP account with AI Core entitlement
- Docker image of the model (see `Dockerfile` in repo root)
- SAP AI Launchpad access

### Steps

1. **Build and push the Docker image**:
   ```bash
   docker build -t financebench-llm:latest .
   docker tag financebench-llm:latest <your-registry>/financebench-llm:latest
   docker push <your-registry>/financebench-llm:latest
   ```

2. **Create an AI Core serving configuration**:
   ```yaml
   # serving-config.yaml
   apiVersion: ai.sap.com/v1alpha1
   kind: ServingTemplate
   metadata:
     name: financebench-llm
     labels:
       scenarios.ai.sap.com/id: "financebench"
       ai.sap.com/version: "1.0.0"
   spec:
     template:
       apiVersion: serving.kserve.io/v1beta1
       spec:
         predictor:
           containers:
             - name: financebench-llm
               image: <your-registry>/financebench-llm:latest
               ports:
                 - containerPort: 7860
                   protocol: TCP
               env:
                 - name: HF_TOKEN
                   valueFrom:
                     secretKeyRef:
                       name: hf-credentials
                       key: token
   ```

3. **Deploy via SAP AI Launchpad**:
   - Navigate to ML Operations > Configurations
   - Create new configuration with the serving template
   - Start deployment
   - Monitor via SAP AI Launchpad dashboard

### Model Versioning

Use SAP AI Core's model versioning to track LoRA adapter iterations:

| Version | Training Data | Exact Match | F1 Score | Status |
|---------|---------------|-------------|----------|--------|
| v1.0    | FinanceBench (120 examples) | 0.52 | 0.71 | Production |
| v1.1    | + Internal financial QA | TBD | TBD | Staging |

---

## 2. SAP HANA Cloud Vector Engine for RAG

SAP HANA Cloud's vector engine enables storing and querying document embeddings for retrieval-augmented generation.

### Setup HANA Vector Store

```sql
-- Create table for SEC filing embeddings
CREATE TABLE FINANCEBENCH_EMBEDDINGS (
    ID BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    DOC_NAME NVARCHAR(500),
    CHUNK_TEXT NCLOB,
    CHUNK_INDEX INTEGER,
    FILING_TYPE NVARCHAR(10),  -- '10-K', '10-Q'
    COMPANY_CODE NVARCHAR(20),
    FISCAL_YEAR INTEGER,
    EMBEDDING REAL_VECTOR(768),  -- Embedding dimension
    CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create vector similarity index
CREATE VECTOR INDEX IDX_FINANCEBENCH_EMBED
ON FINANCEBENCH_EMBEDDINGS(EMBEDDING)
USING HNSW;
```

### Query Pattern

```sql
-- Find similar documents for a financial question
SELECT TOP 5
    DOC_NAME,
    CHUNK_TEXT,
    COSINE_SIMILARITY(EMBEDDING, TO_REAL_VECTOR(:query_embedding)) AS similarity
FROM FINANCEBENCH_EMBEDDINGS
WHERE COSINE_SIMILARITY(EMBEDDING, TO_REAL_VECTOR(:query_embedding)) > 0.7
ORDER BY similarity DESC;
```

### Integration with Python

```python
from hdbcli import dbapi

conn = dbapi.connect(
    address="<hana-host>.hanacloud.ondemand.com",
    port=443,
    user="<user>",
    password="<password>",
    encrypt=True,
)

cursor = conn.cursor()
cursor.execute("""
    SELECT TOP 5 CHUNK_TEXT
    FROM FINANCEBENCH_EMBEDDINGS
    ORDER BY COSINE_SIMILARITY(EMBEDDING, TO_REAL_VECTOR(?)) DESC
""", (query_embedding,))

context_chunks = [row[0] for row in cursor.fetchall()]
```

---

## 3. SAP S/4HANA Financial Data Integration

Connect to SAP S/4HANA for real-time enterprise financial data via OData APIs.

### Available OData Services

| Service | Endpoint | Data |
|---------|----------|------|
| Financial Statements | `API_FINANCIAL_STATEMENT_SRV` | P&L, Balance Sheet |
| Journal Entries | `API_JOURNAL_ENTRY_SRV` | Individual postings |
| Cost Centers | `API_COSTCENTER_SRV` | Cost allocation |
| Profit Centers | `API_PROFITCENTER_SRV` | Profitability data |

### Authentication

Use SAP BTP Destination Service for secure OAuth2 connectivity:

```yaml
# BTP Destination Configuration
Name: S4HANA_FINANCIAL
Type: HTTP
URL: https://<s4hana-host>/sap/opu/odata/sap/
Authentication: OAuth2SAMLBearerAssertion
ProxyType: Internet
```

### Data Flow

1. **Fetch financial data** from S/4HANA via OData
2. **Format as context** for the QA model (structured financial statements)
3. **Combine with SEC filing context** from HANA vector store
4. **Generate answer** using the LoRA fine-tuned model via NIM

This enables questions like:
- *"How does our internal Q3 revenue compare to Apple's reported Q3 revenue?"*
- *"What is our operating margin vs. the industry average from SEC filings?"*

---

## 4. Expose via SAP BTP

### Option A: Kyma Runtime (Kubernetes)

```yaml
# kyma-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: financebench-llm
  namespace: default
spec:
  replicas: 1
  selector:
    matchLabels:
      app: financebench-llm
  template:
    spec:
      containers:
        - name: app
          image: <your-registry>/financebench-llm:latest
          ports:
            - containerPort: 7860
          env:
            - name: HF_TOKEN
              valueFrom:
                secretKeyRef:
                  name: hf-credentials
                  key: token
---
apiVersion: v1
kind: Service
metadata:
  name: financebench-llm
spec:
  selector:
    app: financebench-llm
  ports:
    - port: 80
      targetPort: 7860
---
apiVersion: gateway.kyma-project.io/v1beta1
kind: APIRule
metadata:
  name: financebench-llm
spec:
  gateway: kyma-gateway.kyma-system.svc.cluster.local
  host: financebench-llm.<cluster-domain>
  service:
    name: financebench-llm
    port: 80
  rules:
    - path: /.*
      methods: ["GET", "POST"]
      accessStrategies:
        - handler: jwt
```

### Option B: Cloud Foundry

```yaml
# manifest.yml
applications:
  - name: financebench-llm
    memory: 2G
    instances: 1
    docker:
      image: <your-registry>/financebench-llm:latest
    env:
      HF_TOKEN: ((hf_token))
    routes:
      - route: financebench-llm.cfapps.<landscape>.hana.ondemand.com
```

---

## 5. Compliance & Governance

### SOX Compliance

For Sarbanes-Oxley compliance in financial AI applications:

| Requirement | Implementation |
|-------------|----------------|
| **Audit trail** | MLflow logs every model prediction with timestamps, model version, and input data hash |
| **Change management** | LoRA adapter versions tracked in Git + SAP AI Core model registry |
| **Access control** | SAP BTP role-based access (Business User, Data Scientist, Admin) |
| **Data lineage** | Training data provenance tracked: PatronusAI/financebench → preprocessing → LoRA training |
| **Model validation** | Automated evaluation pipeline (EM, F1, LLM-as-Judge) runs on every adapter version |

### Data Residency

| Deployment | Data Location | Suitable For |
|------------|---------------|--------------|
| SAP AI Core (EU) | EU data centers | GDPR-compliant EU operations |
| On-prem NVIDIA NIM | Customer infrastructure | Maximum data sovereignty |
| HF Spaces (US) | US data centers | Public demo only |

### Model Governance Checklist

- [ ] Model card documents training data, methodology, and limitations
- [ ] Evaluation metrics logged to MLflow for every adapter version
- [ ] No PII in training data (FinanceBench uses public SEC filings)
- [ ] Model outputs include disclaimer: "Not investment advice"
- [ ] Regular re-evaluation on new financial quarters

---

## 6. SAP Generative AI Hub Integration

SAP's Generative AI Hub (part of SAP AI Core) provides:

- **Prompt management**: Version and manage financial QA prompts
- **Model orchestration**: Route between LoRA adapters based on question type
- **Content filtering**: Built-in guardrails for financial content
- **Usage analytics**: Track query volumes, response quality, user satisfaction

### Connection Pattern

```
User Query → SAP AI Launchpad → Generative AI Hub → LoRA Model (NIM) → Response
                                       ↓
                              HANA Vector Store (RAG context)
                                       ↓
                              S/4HANA (ERP financial data)
```

---

## Architecture Decision Records

### ADR-001: Why NVIDIA NIM + SAP AI Core

**Decision**: Use NVIDIA NIM for model inference, deployed on SAP AI Core.

**Rationale**:
- NIM provides optimized inference with multi-LoRA support
- SAP AI Core provides enterprise MLOps (versioning, monitoring, scaling)
- Combined stack gives both performance and governance

**Alternatives considered**:
- Pure SAP AI Core with custom serving → slower inference
- Pure NVIDIA NIM on bare metal → no enterprise governance
- OpenAI API → data residency concerns for financial data

### ADR-002: Why SAP HANA Cloud for Vector Store

**Decision**: Use HANA Cloud Vector Engine instead of standalone vector DB (Pinecone, Weaviate).

**Rationale**:
- Already in SAP ecosystem — no additional vendor
- Transactional + analytical + vector in one database
- Financial data already in HANA (from S/4HANA)
- Enterprise security and compliance built-in

### ADR-003: Why LoRA over Full Fine-tuning

**Decision**: Use LoRA (r=16) instead of full model fine-tuning.

**Rationale**:
- 120 training examples insufficient for full fine-tuning
- LoRA adapter is ~50MB vs 16GB full model — faster deployment
- Multi-LoRA NIM enables domain-specific adapters without model duplication
- Can be swapped at inference time (e.g., finance vs. insurance adapter)
