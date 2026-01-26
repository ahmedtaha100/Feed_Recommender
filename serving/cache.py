import json
import pickle
from typing import Optional, List, Dict, Any
import logging

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RedisCache:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        embedding_ttl: int = 3600,
        max_connections: int = 100,
    ):
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.embedding_ttl = embedding_ttl
        self.max_connections = max_connections
        self.client = None
        self._connected = False

    def connect(self) -> bool:
        try:
            import redis
            self.client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                max_connections=self.max_connections,
                decode_responses=False,
            )
            self.client.ping()
            self._connected = True
            logger.info(f"Connected to Redis at {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}. Using in-memory fallback.")
            self._connected = False
            self._memory_cache: Dict[str, Any] = {}
            return False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def _get_embedding_key(self, user_id: str) -> str:
        return f"user_emb:{user_id}"

    def _get_history_key(self, user_id: str) -> str:
        return f"user_hist:{user_id}"

    def get_user_embedding(self, user_id: str) -> Optional[np.ndarray]:
        key = self._get_embedding_key(user_id)

        if self._connected:
            try:
                data = self.client.get(key)
                if data:
                    return pickle.loads(data)
            except Exception as e:
                logger.error(f"Redis get error: {e}")
        else:
            return self._memory_cache.get(key)

        return None

    def set_user_embedding(self, user_id: str, embedding: np.ndarray) -> bool:
        key = self._get_embedding_key(user_id)

        if self._connected:
            try:
                self.client.setex(key, self.embedding_ttl, pickle.dumps(embedding))
                return True
            except Exception as e:
                logger.error(f"Redis set error: {e}")
                return False
        else:
            self._memory_cache[key] = embedding
            return True

    def get_user_history(self, user_id: str) -> Optional[List[str]]:
        key = self._get_history_key(user_id)

        if self._connected:
            try:
                data = self.client.get(key)
                if data:
                    return json.loads(data.decode())
            except Exception as e:
                logger.error(f"Redis get error: {e}")
        else:
            return self._memory_cache.get(key)

        return None

    def set_user_history(self, user_id: str, history: List[str]) -> bool:
        key = self._get_history_key(user_id)

        if self._connected:
            try:
                self.client.setex(key, self.embedding_ttl, json.dumps(history))
                return True
            except Exception as e:
                logger.error(f"Redis set error: {e}")
                return False
        else:
            self._memory_cache[key] = history
            return True

    def delete_user_embedding(self, user_id: str) -> bool:
        key = self._get_embedding_key(user_id)

        if self._connected:
            try:
                self.client.delete(key)
                return True
            except Exception as e:
                logger.error(f"Redis delete error: {e}")
                return False
        else:
            self._memory_cache.pop(key, None)
            return True

    def get_batch_embeddings(self, user_ids: List[str]) -> Dict[str, Optional[np.ndarray]]:
        results = {}

        if self._connected:
            try:
                keys = [self._get_embedding_key(uid) for uid in user_ids]
                values = self.client.mget(keys)
                for user_id, value in zip(user_ids, values):
                    if value:
                        results[user_id] = pickle.loads(value)
                    else:
                        results[user_id] = None
            except Exception as e:
                logger.error(f"Redis mget error: {e}")
                for user_id in user_ids:
                    results[user_id] = None
        else:
            for user_id in user_ids:
                results[user_id] = self._memory_cache.get(self._get_embedding_key(user_id))

        return results

    def set_batch_embeddings(self, embeddings: Dict[str, np.ndarray]) -> bool:
        if self._connected:
            try:
                pipe = self.client.pipeline()
                for user_id, embedding in embeddings.items():
                    key = self._get_embedding_key(user_id)
                    pipe.setex(key, self.embedding_ttl, pickle.dumps(embedding))
                pipe.execute()
                return True
            except Exception as e:
                logger.error(f"Redis pipeline error: {e}")
                return False
        else:
            for user_id, embedding in embeddings.items():
                self._memory_cache[self._get_embedding_key(user_id)] = embedding
            return True

    def clear_all(self) -> bool:
        if self._connected:
            try:
                self.client.flushdb()
                return True
            except Exception as e:
                logger.error(f"Redis flushdb error: {e}")
                return False
        else:
            self._memory_cache.clear()
            return True

    def get_stats(self) -> Dict[str, Any]:
        if self._connected:
            try:
                info = self.client.info()
                return {
                    "connected": True,
                    "used_memory": info.get("used_memory_human", "unknown"),
                    "connected_clients": info.get("connected_clients", 0),
                    "total_keys": self.client.dbsize(),
                }
            except Exception as e:
                return {"connected": False, "error": str(e)}
        else:
            return {
                "connected": False,
                "mode": "in-memory",
                "total_keys": len(self._memory_cache),
            }
