# Two-Tower Feed Recommender

A production-grade two-stage feed recommendation system built with PyTorch, FAISS, and FastAPI.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Two-Tower Model                          │
├────────────────────────────┬────────────────────────────────────┤
│        User Tower          │           Item Tower               │
│                            │                                    │
│  ┌──────────────────┐      │      ┌──────────────────┐          │
│  │  History Items   │      │      │   Title Encoder  │          │
│  │  (Attention)     │      │      │   (Self-Attn)    │          │
│  └────────┬─────────┘      │      └────────┬─────────┘          │
│           │                │               │                    │
│  ┌────────▼─────────┐      │      ┌────────▼─────────┐          │
│  │  User Features   │      │      │ Abstract Encoder │          │
│  │   Projection     │      │      │   (Self-Attn)    │          │
│  └────────┬─────────┘      │      └────────┬─────────┘          │
│           │                │               │                    │
│           │                │      ┌────────▼─────────┐          │
│           │                │      │ Category Embed   │          │
│           │                │      └────────┬─────────┘          │
│           │                │               │                    │
│  ┌────────▼─────────┐      │      ┌────────▼─────────┐          │
│  │  L2 Normalize    │      │      │  L2 Normalize    │          │
│  │  128-dim embed   │      │      │  128-dim embed   │          │
│  └────────┬─────────┘      │      └────────┬─────────┘          │
│           │                │               │                    │
└───────────┼────────────────┴───────────────┼────────────────────┘
            │                                │
            │    ┌───────────────────┐       │
            └────►  InfoNCE Loss     ◄───────┘
                 │  (temp=0.07)      │
                 └───────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Retrieval Stage                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │ User Embed   │───►│  FAISS IVF   │───►│ Top-K Candidates │   │
│  │              │    │  Index       │    │  (<15ms latency) │   │
│  └──────────────┘    └──────────────┘    └──────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Reranking Stage                             │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                   LightGBM Reranker                      │   │
│  │  Features: similarity, popularity, category match, etc.  │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Serving Layer                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │   FastAPI    │◄──►│ Redis Cache  │    │  Async Workers   │   │
│  │   Server     │    │ (User Embeds)│    │                  │   │
│  └──────────────┘    └──────────────┘    └──────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Features

- Two-Tower Retrieval Model: Separate user and item encoders with shared embedding space
- InfoNCE Contrastive Loss: Temperature-scaled with in-batch and sampled negatives
- FAISS ANN Search: IVF index for sub-15ms retrieval on 1M+ items
- LightGBM Reranker: Second-stage ranking with rich features
- FastAPI + Redis: Production-ready serving with caching
- Airflow Pipeline: Daily batch processing with drift detection

## Quickstart

### Installation

```bash
pip install -r requirements.txt
```

### Run Demo

```bash
python run_demo.py
```

This will:
1. Download the MIND dataset
2. Preprocess the data
3. Train the two-tower model
4. Build the FAISS index
5. Run evaluation and benchmarks
6. Show demo recommendations

### Start API Server

```bash
python -m serving.app --port 8000
```

### API Usage

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "history": ["N12345", "N67890"],
    "num_recommendations": 20
  }'
```

## Project Structure

```
Feed_Recommender/
├── config/
│   └── config.yaml          # All hyperparameters
├── data/
│   ├── download.py          # MIND dataset downloader
│   ├── preprocessing.py     # Data preprocessing
│   └── dataset.py           # PyTorch datasets
├── models/
│   ├── towers.py            # User and Item towers
│   ├── two_tower.py         # Main model with loss
│   └── reranker.py          # LightGBM reranker
├── retrieval/
│   ├── index_builder.py     # FAISS index building
│   └── retriever.py         # ANN search interface
├── serving/
│   ├── app.py               # FastAPI application
│   ├── cache.py             # Redis caching layer
│   └── schemas.py           # Pydantic models
├── pipeline/
│   ├── dag.py               # Airflow DAG
│   └── drift_detection.py   # Data drift validation
├── evaluation/
│   ├── metrics.py           # Recall@K, NDCG@K, MRR
│   └── benchmark.py         # Latency benchmarks
├── training/
│   ├── trainer.py           # Training loop
│   └── train.py             # Training script
├── tests/
│   ├── test_models.py
│   ├── test_retrieval.py
│   └── test_serving.py
├── requirements.txt
├── run_demo.py              # End-to-end demo
└── README.md
```

## Results

### Evaluation Metrics (MINDsmall)

| Metric      | Value  |
|-------------|--------|
| Recall@10   | 0.15+  |
| Recall@50   | 0.35+  |
| Recall@100  | 0.50+  |
| NDCG@10     | 0.10+  |
| MRR         | 0.08+  |

### Latency Benchmarks

| Component          | Mean (ms) | P99 (ms) |
|--------------------|-----------|----------|
| FAISS Retrieval    | <5        | <15      |
| Model Inference    | <10       | <20      |
| End-to-End API     | <30       | <50      |

## Configuration

Key parameters in `config/config.yaml`:

```yaml
model:
  embedding_dim: 128
  word_embedding_dim: 300
  attention_heads: 8
  dropout: 0.2

two_tower:
  temperature: 0.07
  use_in_batch_negatives: true
  num_sampled_negatives: 4

training:
  batch_size: 256
  learning_rate: 0.001
  num_epochs: 10

faiss:
  index_type: IVF
  nlist: 100
  nprobe: 10
```

## Development

### Run Tests

```bash
pytest tests/ -v
```

### Train Model

```bash
python -m training.train --config config/config.yaml
```

### Build Index

```bash
python -m retrieval.index_builder \
  --embeddings artifacts/item_embeddings.npy \
  --output artifacts/faiss_index
```

### Run Evaluation

```bash
python -m evaluation.metrics \
  --model artifacts/two_tower_model.pt \
  --index artifacts/faiss_index
```

### Run Benchmarks

```bash
python -m evaluation.benchmark
```

## Dataset

This project uses the Microsoft MIND (Microsoft News Dataset) for training and evaluation.

- MINDsmall: ~160k users, ~50k news articles
- Download: https://msnews.github.io/

## License

MIT License
