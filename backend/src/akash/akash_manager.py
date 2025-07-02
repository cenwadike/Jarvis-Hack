import os
import subprocess
import json
from cosmpy.aerial.client import LedgerClient
from cosmpy.aerial.wallet import LocalWallet
from cosmpy.aerial.config import NetworkConfig
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from dotenv import find_dotenv, load_dotenv
import yaml

# Explicitly specify the .env file path
dotenv_path = find_dotenv(usecwd=True)
loaded = load_dotenv(dotenv_path)

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
Base = declarative_base()

class Deployment(Base):
    __tablename__ = "deployments"
    id = Column(Integer, primary_key=True)
    deployment_id = Column(String(255), unique=True)  # DSeq
    tx_hash = Column(String(255))
    wallet_address = Column(String(64), nullable=False)
    status = Column(String(50))
    image = Column(String(100))
    cpu = Column(Float)
    memory = Column(String(20))
    storage = Column(String(20))
    ports = Column(Text)  # JSON string
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

# Akash testnet configuration for cosmpy
AKASH_TESTNET_CONFIG = NetworkConfig(
    chain_id=os.getenv("AKASH_CHAIN_ID"),
    url=f"grpc+{os.getenv('AKASH_NODE')}",
    fee_minimum_gas_price=1,
    fee_denomination="uakt",
    staking_denomination="uakt"
)
client = LedgerClient(AKASH_TESTNET_CONFIG)

wallet = LocalWallet.from_mnemonic(os.getenv("AKASH_MNEMONIC"))
ADDRESS = str(wallet.address())

def generate_sdl(image, cpu, memory, storage, ports):
    """Generate a dynamic SDL with compute, storage, and networking."""
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

def deploy_to_akash(wallet_address, image="nginx", cpu=0.1, memory="512Mi", storage="512Mi", ports=["80"]):
    session = Session()
    sdl_content = generate_sdl(image, cpu, memory, storage, ports)
    
    # Write SDL to a temporary file
    with open("temp_sdl.yml", "w") as f:
        yaml.dump(sdl_content, f)
    
    # Use Akash CLI (provider-services) for deployment
    cmd = [
        "provider-services", "tx", "deployment", "create", "temp_sdl.yml",
        "--from", "my_wallet",
        "--chain-id", os.environ["AKASH_CHAIN_ID"],
        "--node", os.environ["AKASH_NODE"],
        "--keyring-backend", os.environ["AKASH_KEYRING_BACKEND"],
        "--gas", "auto",
        "--yes"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, env=os.environ)
    if result.returncode != 0:
        raise Exception(f"Deployment failed: {result.stderr}")
    
    # Parse tx output to get txhash and dseq
    tx_output = json.loads(result.stdout)
    tx_hash = tx_output["txhash"]
    dseq = tx_output["logs"][0]["events"][0]["attributes"][0]["value"]  # Adjust based on actual output
    
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
    session.add(deployment)
    session.commit()
    session.close()
    os.remove("temp_sdl.yml")
    return dseq

def get_deployment_status(deployment_id):
    session = Session()
    deployment = session.query(Deployment).filter_by(deployment_id=deployment_id).first()
    if not deployment:
        session.close()
        return "Not found"
    
    # Query Akash network for real status
    try:
        status_response = client.query_deployment(deployment_id, deployment.wallet_address)
        status = "active" if status_response["state"] == "active" else status_response["state"]
        deployment.status = status
        session.commit()
    except Exception as e:
        print(f"Failed to query status: {e}")
        status = deployment.status  # Fallback to DB status
    session.close()
    return status

def terminate_deployment(deployment_id):
    session = Session()
    deployment = session.query(Deployment).filter_by(deployment_id=deployment_id).first()
    if not deployment:
        session.close()
        return False
    
    # Close deployment on Akash
    try:
        tx = client.close_deployment(deployment_id, deployment.wallet_address, wallet)
        tx_result = client.broadcast_tx(tx, wallet)
        if tx_result["code"] == 0:
            deployment.status = "closed"
            session.commit()
            session.close()
            return True
        else:
            print(f"Termination failed: {tx_result['raw_log']}")
    except Exception as e:
        print(f"Termination error: {e}")
    session.close()
    return False
def get_all_deployments(wallet_address, page=1, per_page=10):
    session = Session()
    # Paginate deployments for the specific wallet_address
    query = session.query(Deployment).filter_by(wallet_address=wallet_address)
    total = query.count()
    deployments = query.order_by(Deployment.created_at.desc())\
                      .offset((page - 1) * per_page)\
                      .limit(per_page).all()
    
    result = [
        {
            "id": d.deployment_id,
            "tx_hash": d.tx_hash,
            "status": d.status,
            "image": d.image,
            "cpu": d.cpu,
            "memory": d.memory,
            "storage": d.storage,
            "ports": d.ports,
            "created_at": str(d.created_at)
        } for d in deployments
    ]
    session.close()
    
    return {
        "deployments": result,
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": (total + per_page - 1) // per_page
    }