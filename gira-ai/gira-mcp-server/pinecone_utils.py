"""Utility helpers for safely inspecting and deleting Pinecone namespaces/index data.

Usage:
  export PINECONE_API_KEY=...
  export PINECONE_ENV=...
  export PINECONE_INDEX=quickstart-py
  python pinecone_utils.py --list-namespaces
  python pinecone_utils.py --delete-namespace FjgiKTuJhNAU

This uses the public `pinecone-client` package (pip install pinecone-client).
"""

import os
import sys
import json
import argparse
import logging

try:
    import pinecone
except Exception as e:
    raise RuntimeError("pinecone client library is required. install with: pip install pinecone-client") from e

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_pinecone():
    api_key = os.getenv("PINECONE_API_KEY")
    env = os.getenv("PINECONE_ENV") or os.getenv("PINECONE_ENVIRONMENT") or os.getenv("PINECONE_REGION")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY environment variable is not set")
    if not env:
        raise RuntimeError("PINECONE_ENV (environment/region) environment variable is not set")

    pinecone.init(api_key=api_key, environment=env)


def get_index(index_name: str):
    if not index_name:
        raise ValueError("index_name must be provided")
    return pinecone.Index(index_name)


def list_namespaces(index_name: str):
    """Return list of namespaces present in the index (may be empty)."""
    index = get_index(index_name)
    stats = index.describe_index_stats()
    namespaces = list(stats.get("namespaces", {}).keys())
    return namespaces


def safe_delete_namespace(index_name: str, namespace: str, dry_run: bool = True) -> dict:
    """Delete all vectors in `namespace` if it exists. Returns a result dict.

    If namespace does not exist this is not an error — we'll return a message and no-op.
    Use dry_run=False to actually delete.
    """
    if not namespace:
        raise ValueError("namespace must be provided")

    index = get_index(index_name)

    try:
        stats = index.describe_index_stats(namespace=namespace)
        namespaces = stats.get("namespaces", {})
        if namespace not in namespaces and not namespaces:
            # Some Pinecone versions return empty dict when namespace not found
            logger.info(f"Namespace '{namespace}' not present in index '{index_name}' (no-op)")
            return {"ok": True, "deleted": False, "reason": "namespace_not_found"}

        # Show counts for confirmation
        ns_info = namespaces.get(namespace, {})
        vector_count = ns_info.get("vector_count") if ns_info else None
        logger.info(f"Namespace '{namespace}' found in index '{index_name}' (vectors={vector_count})")

        if dry_run:
            logger.info("Dry run enabled - not deleting. Pass dry_run=False to actually delete.")
            return {"ok": True, "deleted": False, "dry_run": True, "vector_count": vector_count}

        # Perform deletion of all vectors in namespace
        logger.info(f"Deleting namespace '{namespace}' in index '{index_name}'... This may take a while")
        index.delete(delete_all=True, namespace=namespace)
        logger.info("Delete request submitted")
        return {"ok": True, "deleted": True, "namespace": namespace}

    except Exception as e:
        # Pinecone client raises generic Exception for 404 in many versions; inspect message
        msg = str(e)
        if "Namespace not found" in msg or "404" in msg or "Not Found" in msg:
            logger.warning(f"Namespace '{namespace}' not found (treated as no-op): {msg}")
            return {"ok": True, "deleted": False, "reason": "namespace_not_found", "error": msg}
        logger.exception("Unexpected error while deleting namespace")
        return {"ok": False, "error": msg}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default=os.getenv("PINECONE_INDEX", "quickstart-py"), help="Pinecone index name")
    parser.add_argument("--list-namespaces", action="store_true")
    parser.add_argument("--delete-namespace", type=str, help="Namespace to delete")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Don't actually delete (default)")

    args = parser.parse_args()

    init_pinecone()

    if args.list_namespaces:
        ns = list_namespaces(args.index)
        print(json.dumps({"namespaces": ns}, indent=2))
        sys.exit(0)

    if args.delete_namespace:
        res = safe_delete_namespace(args.index, args.delete_namespace, dry_run=args.dry_run)
        print(json.dumps(res, indent=2))
        sys.exit(0)

    parser.print_help()