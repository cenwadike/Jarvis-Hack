import os
import json
import tempfile
import yaml
from cosmpy.aerial.client import LedgerClient
from cosmpy.aerial.wallet import LocalWallet
from cosmpy.aerial.config import NetworkConfig
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from dotenv import find_dotenv, load_dotenv
import re
import logging
from typing import List, Dict, Any, Optional
from contextlib import contextmanager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
dotenv_path = find_dotenv(usecwd=True)
load_dotenv(dotenv_path)

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set")
engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=5, max_overflow=10)
Base = declarative_base()

class Deployment(Base):
    __tablename__ = "deployments"
    id = Column(Integer, primary_key=True)
    deployment_id = Column(String(255), unique=True, nullable=False)
    tx_hash = Column(String(255))
    wallet_address = Column(String(64), nullable=False, index=True)
    status = Column(String(50), nullable=False, default="pending")
    image = Column(String(100), nullable=False)
    cpu = Column(Float, nullable=False)
    memory = Column(String(20), nullable=False)
    storage = Column(String(20), nullable=False)
    ports = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine, expire_on_commit=False)

# Akash configuration
AKASH_TESTNET_CONFIG = NetworkConfig(
    chain_id=os.getenv("AKASH_CHAIN_ID", "akashnet-2"),
    url=f"grpc+{os.getenv('AKASH_NODE', 'http://akash-node:26657')}",
    fee_minimum_gas_price=1,
    fee_denomination="uakt",
    staking_denomination="uakt"
)
client = LedgerClient(AKASH_TESTNET_CONFIG)

# Input validation patterns
IMAGE_PATTERN = re.compile(r'^[a-zA-Z0-9._/-]+:[a-zA-Z0-9._-]+$')
MEMORY_STORAGE_PATTERN = re.compile(r'^\d+(Mi|Gi|Ti)$')
PORT_PATTERN = re.compile(r'^\d+$')

wallet = LocalWallet.from_mnemonic(os.getenv("AKASH_MNEMONIC")) # type: ignore

@contextmanager
def session_scope():
    """Provide a transactional scope around a series of operations."""
    session = Session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

def validate_deployment_params(image: str, cpu: float, memory: str, storage: str, ports: List[str]) -> None:
    """Validate deployment parameters."""
    if not IMAGE_PATTERN.match(image):
        raise ValueError(f"Invalid Docker image format: {image}")
    if not isinstance(cpu, (int, float)) or cpu <= 0 or cpu > 100:
        raise ValueError(f"Invalid CPU value: {cpu}. Must be between 0 and 100.")
    if not MEMORY_STORAGE_PATTERN.match(memory):
        raise ValueError(f"Invalid memory format: {memory}. Must be e.g., '512Mi'.")
    if not MEMORY_STORAGE_PATTERN.match(storage):
        raise ValueError(f"Invalid storage format: {storage}. Must be e.g., '512Mi'.")
    for port in ports:
        if not PORT_PATTERN.match(port) or not (1 <= int(port) <= 65535):
            raise ValueError(f"Invalid port: {port}. Must be a number between 1 and 65535.")

def generate_sdl(image: str, cpu: float, memory: str, storage: str, ports: List[str]) -> Dict[str, Any]:
    """Generate a dynamic SDL with compute, storage, and networking."""
    validate_deployment_params(image, cpu, memory, storage, ports)
    sdl = {
        "version": "2.0",
        "services": {
            "app": {
                "image": image,
                "count": 1,
                "expose": [
                    {"port": int(port), "as": int(port), "to": [{"global": True}]} for port in ports
                ]
            }
        },
        "profiles": {
            "compute": {
                "app": {
                    "resources": {
                        "cpu": {"units": cpu},
                        "memory": {"size": memory},
                        "storage": [{"size": storage}]
                    }
                }
            },
            "placement": {
                "akash": {
                    "pricing": {
                        "app": {
                            "denom": "uakt",
                            "amount": 1000
                        }
                    }
                }
            }
        },
        "deployment": {
            "app": {
                "akash": {
                    "profile": "app",
                    "count": 1
                }
            }
        }
    }
    return sdl

def deploy_to_akash(
    wallet: LocalWallet,
    wallet_address: str,
    image: str = "nginx",
    cpu: float = 0.1,
    memory: str = "512Mi",
    storage: str = "512Mi",
    ports: List[str] = ["80"],
    db_session: Optional[Any] = None
) -> str:
    """Deploy to Akash network securely."""
    try:
        validate_deployment_params(image, cpu, memory, storage, ports)
        sdl_content = generate_sdl(image, cpu, memory, storage, ports)

        # Use secure temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as temp_file:
            yaml.safe_dump(sdl_content, temp_file)
            temp_file_path = temp_file.name

        try:
            # Use cosmpy for deployment instead of CLI for better security
            tx = client.create_deployment(sdl_content, wallet_address, wallet)
            tx_result = client.broadcast_tx(tx, wallet) # type: ignore
            if tx_result["code"] != 0:
                raise Exception(f"Deployment failed: {tx_result['raw_log']}")

            dseq = tx_result["logs"][0]["events"][0]["attributes"][0]["value"]
            tx_hash = tx_result["txhash"]

            deployment = Deployment(
                deployment_id=dseq,
                tx_hash=tx_hash,
                wallet_address=wallet_address,
                status="pending",
                image=image,
                cpu=cpu,
                memory=memory,
                storage=storage,
                ports=json.dumps(ports)
            )

            if db_session:
                db_session.add(deployment)
                db_session.flush()  # Let the test handle commit/rollback
            else:
                with session_scope() as session:
                    session.add(deployment)
                    session.commit()

            logger.info(f"Deployment created with ID: {dseq}")
            return dseq
        finally:
            os.unlink(temp_file_path)
    except Exception as e:
        logger.error(f"Deployment error for {wallet_address}: {str(e)}")
        raise

def get_deployment_status(
    wallet_address: str,
    deployment_id: str,
    db_session: Optional[Any] = None
) -> str:
    """Query the status of a deployment on the Akash network.

    Args:
        wallet_address: Akash wallet address.
        deployment_id: Deployment ID (dseq).
        db_session: Optional SQLAlchemy session for testing.

    Returns:
        Deployment status (e.g., 'pending', 'active', 'closed', 'Not found', 'error').

    Raises:
        ValueError: If deployment_id format is invalid.
    """
    if not re.match(r'^[a-zA-Z0-9_-]{1,50}$', deployment_id):
        raise ValueError("Invalid deployment ID format")

    session = db_session or Session()
    try:
        deployment = session.query(Deployment).filter_by(
            deployment_id=deployment_id,
            wallet_address=wallet_address
        ).first()

        if not deployment:
            logger.warning(f"Deployment {deployment_id} not found for wallet: {wallet_address}")
            return "Not found"

        try:
            status_response = client.query_deployment_status(wallet_address, deployment_id)
            status = status_response.get("state", "error")
            if status == "active":
                status = "active"
            elif status not in ["pending", "closed"]:
                status = "error"

            deployment.status = status
            if not db_session:
                session.commit()
            else:
                session.flush()

            logger.info(f"Deployment {deployment_id} status updated to {status} for wallet: {wallet_address}")
            return status

        except Exception as e:
            logger.warning(f"Failed to query status for {deployment_id}: {str(e)}")
            return deployment.status if deployment else "error"

    except Exception as e:
        logger.error(f"Database error querying status for {deployment_id}: {str(e)}")
        return "error"

    finally:
        if not db_session:
            session.close()

def terminate_deployment(
    wallet_address: str,
    deployment_id: str,
    wallet: LocalWallet,
    db_session: Optional[Session] = None # type: ignore
) -> bool:
    """Terminate a deployment with authorization check and optional session injection."""
    if not re.match(r'^[a-zA-Z0-9_-]{1,50}$', deployment_id):
        raise ValueError("Invalid deployment ID format")

    session = db_session or session_scope().__enter__()
    try:
        deployment = session.query(Deployment).filter_by(
            deployment_id=deployment_id,
            wallet_address=wallet_address
        ).first()
        if not deployment:
            logger.warning(f"Deployment {deployment_id} not found for wallet: {wallet_address}")
            return False

        tx = client.close_deployment(deployment_id, wallet_address, wallet)
        tx_result = client.broadcast_tx(tx, wallet)

        if tx_result.get("code", 1) == 0:
            deployment.status = "closed"
            if db_session:
                session.flush()
            else:
                session.commit()
            logger.info(f"Deployment {deployment_id} terminated successfully")
            return True
        else:
            logger.error(f"Termination failed for {deployment_id}: {tx_result.get('raw_log')}")
            return False

    except Exception as e:
        logger.error(f"Termination error for {deployment_id}: {str(e)}")
        return False

    finally:
        if not db_session:
            session.close()

def get_all_deployments(
    wallet_address: str,
    page: int = 1,
    per_page: int = 10,
    db_session: Optional[Session] = None # type: ignore
) -> Dict[str, Any]:
    """Get paginated deployments for a wallet address."""
    if page < 1 or per_page < 1 or per_page > 100:
        raise ValueError("Invalid pagination parameters")

    session = db_session or session_scope().__enter__()
    try:
        query = session.query(Deployment).filter_by(wallet_address=wallet_address)
        total = query.count()
        deployments = (
            query.order_by(Deployment.created_at.desc())
            .offset((page - 1) * per_page)
            .limit(per_page)
            .all()
        )

        if total == 0:
            logger.info(f"No deployments found for {wallet_address}")

        return {
            "deployments": [
                {
                    "id": d.deployment_id,
                    "tx_hash": d.tx_hash,
                    "status": d.status,
                    "image": d.image,
                    "cpu": d.cpu,
                    "memory": d.memory,
                    "storage": d.storage,
                    "ports": json.loads(d.ports),
                    "created_at": d.created_at.isoformat()
                } for d in deployments
            ],
            "total": total,
            "page": page,
            "per_page": per_page,
            "total_pages": (total + per_page - 1) // per_page
        }
    finally:
        if not db_session:
            session.close()
            