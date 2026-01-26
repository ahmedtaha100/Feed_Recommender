from datetime import datetime, timedelta
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from airflow.operators.bash import BashOperator
    from airflow.utils.dates import days_ago
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False
    logger.warning("Airflow not installed. DAG will not be functional.")


def check_data_freshness(**context):
    import os
    data_dir = Path("data/mind")

    if not data_dir.exists():
        raise ValueError("Data directory not found")

    train_behaviors = data_dir / "MINDsmall_train" / "behaviors.tsv"
    if not train_behaviors.exists():
        raise ValueError("Training data not found")

    mtime = datetime.fromtimestamp(train_behaviors.stat().st_mtime)
    age_days = (datetime.now() - mtime).days

    context['ti'].xcom_push(key='data_age_days', value=age_days)

    logger.info(f"Data age: {age_days} days")
    return age_days


def run_preprocessing(**context):
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from data.preprocessing import MINDPreprocessor

    preprocessor = MINDPreprocessor(
        data_dir="data/mind",
        max_title_len=30,
        max_abstract_len=100,
        max_history_len=50,
        min_word_freq=2,
    )

    stats = preprocessor.preprocess("data/processed")

    context['ti'].xcom_push(key='preprocessing_stats', value=stats)

    logger.info(f"Preprocessing complete: {stats}")
    return stats


def run_drift_detection(**context):
    import sys
    import pickle
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from pipeline.drift_detection import DriftDetector, compute_feature_distributions

    with open("data/processed/train_samples.pkl", "rb") as f:
        samples = pickle.load(f)

    with open("data/processed/news_data.pkl", "rb") as f:
        news_data = pickle.load(f)

    current_features = compute_feature_distributions(samples[:10000], news_data)

    detector = DriftDetector(psi_threshold=0.2)

    reference_path = Path("artifacts/reference_distributions.json")
    if reference_path.exists():
        detector.load_reference(str(reference_path))
        results = detector.detect_all_drift(current_features)

        if results["any_drift_detected"]:
            logger.warning("Data drift detected!")
            for feature, result in results["feature_results"].items():
                if result.get("drift_detected"):
                    logger.warning(f"  {feature}: PSI={result['metrics']['psi']:.4f}")
    else:
        for name, data in current_features.items():
            detector.set_reference(name, data)
        detector.save_reference(str(reference_path))
        results = {"any_drift_detected": False, "message": "Reference data initialized"}

    context['ti'].xcom_push(key='drift_results', value=json.dumps(results, default=str))

    return results


def rebuild_index(**context):
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from retrieval.index_builder import build_index_from_embeddings

    builder = build_index_from_embeddings(
        embeddings_path="artifacts/item_embeddings.npy",
        idx_mapping_path="artifacts/item_idx_mapping.json",
        output_path="artifacts/faiss_index",
        index_type="IVF",
        nlist=100,
        nprobe=10,
    )

    logger.info(f"Index rebuilt with {builder.index.ntotal} items")
    return {"index_size": builder.index.ntotal}


def validate_model(**context):
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from evaluation.metrics import evaluate_retrieval

    results = evaluate_retrieval(
        model_path="artifacts/two_tower_model.pt",
        index_path="artifacts/faiss_index",
        data_path="data/processed",
        top_k=[10, 50, 100],
        num_samples=1000,
    )

    context['ti'].xcom_push(key='validation_results', value=json.dumps(results))

    logger.info(f"Validation results: {results}")
    return results


def send_alerts(**context):
    ti = context['ti']

    drift_results = ti.xcom_pull(task_ids='drift_detection', key='drift_results')
    if drift_results:
        drift_results = json.loads(drift_results)

        if drift_results.get("any_drift_detected"):
            logger.warning("ALERT: Data drift detected! Review required.")

    validation_results = ti.xcom_pull(task_ids='validate_model', key='validation_results')
    if validation_results:
        validation_results = json.loads(validation_results)

        recall_10 = validation_results.get("recall@10", 0)
        if recall_10 < 0.1:
            logger.warning(f"ALERT: Low recall@10: {recall_10:.4f}")

    return {"alerts_sent": True}


if AIRFLOW_AVAILABLE:
    default_args = {
        'owner': 'recommender',
        'depends_on_past': False,
        'email_on_failure': False,
        'email_on_retry': False,
        'retries': 1,
        'retry_delay': timedelta(minutes=5),
    }

    dag = DAG(
        'feed_recommender_pipeline',
        default_args=default_args,
        description='Daily feed recommender pipeline',
        schedule_interval='@daily',
        start_date=days_ago(1),
        catchup=False,
        tags=['recommender', 'ml'],
    )

    check_data = PythonOperator(
        task_id='check_data_freshness',
        python_callable=check_data_freshness,
        dag=dag,
    )

    preprocess = PythonOperator(
        task_id='preprocess_data',
        python_callable=run_preprocessing,
        dag=dag,
    )

    drift_detection = PythonOperator(
        task_id='drift_detection',
        python_callable=run_drift_detection,
        dag=dag,
    )

    rebuild = PythonOperator(
        task_id='rebuild_index',
        python_callable=rebuild_index,
        dag=dag,
    )

    validate = PythonOperator(
        task_id='validate_model',
        python_callable=validate_model,
        dag=dag,
    )

    alerts = PythonOperator(
        task_id='send_alerts',
        python_callable=send_alerts,
        dag=dag,
    )

    check_data >> preprocess >> drift_detection >> rebuild >> validate >> alerts
