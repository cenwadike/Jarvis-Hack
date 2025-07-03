from typing import Optional
import uuid
import pytest
import json
import yaml
from datetime import datetime
from unittest.mock import MagicMock, patch
from sqlalchemy import create_engine # type: ignore
from sqlalchemy.orm import sessionmaker
from src.akash.akash_manager import (
    Deployment, generate_sdl, deploy_to_akash, get_deployment_status, session_scope,
    terminate_deployment, get_all_deployments, validate_deployment_params, Base
)
from src.akash.akash_manager import client
from cosmpy.aerial.wallet import LocalWallet
import logging
logger = logging.getLogger(__name__)

# Test database setup
DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
Base.metadata.create_all(engine)

@pytest.fixture
def session():
    """Provide a fresh database session for each test."""
    session = Session()
    yield session
    session.rollback()
    session.close()

@pytest.fixture
def mock_wallet():
    """Mock a LocalWallet instance."""
    wallet = MagicMock(spec=LocalWallet)
    wallet.address.return_value = "akash1testwalletaddress"
    return wallet

@pytest.fixture
def mock_client():
    """Mock the Akash client."""
    client = MagicMock()
    return client

@pytest.fixture
def caplog(caplog):
    """Capture log messages at DEBUG level."""
    caplog.set_level(logging.DEBUG)
    return caplog

def test_validate_deployment_params_valid():
    """Test valid deployment parameters."""
    validate_deployment_params("nginx:latest", 0.5, "512Mi", "1Gi", ["80", "443"])
    assert True  # No exception raised

def test_validate_deployment_params_invalid():
    """Test invalid deployment parameters."""
    # Invalid Docker image
    with pytest.raises(ValueError, match="Invalid Docker image format"):
        validate_deployment_params("invalid image", 0.5, "512Mi", "1Gi", ["80"])
    # Invalid CPU
    with pytest.raises(ValueError, match="Invalid CPU value"):
        validate_deployment_params("nginx:latest", -0.5, "512Mi", "1Gi", ["80"])
    # Invalid memory
    with pytest.raises(ValueError, match="Invalid memory format"):
        validate_deployment_params("nginx:latest", 0.5, "512MB", "1Gi", ["80"])
    # Invalid storage
    with pytest.raises(ValueError, match="Invalid storage format"):
        validate_deployment_params("nginx:latest", 0.5, "512Mi", "1GB", ["80"])
    # Invalid port
    with pytest.raises(ValueError, match="Invalid port"):
        validate_deployment_params("nginx:latest", 0.5, "512Mi", "1Gi", ["invalid"])

def test_generate_sdl():
    """Test SDL generation for deployment."""
    sdl = generate_sdl("nginx:latest", 0.5, "512Mi", "1Gi", ["80"])
    sdl_dict = yaml.safe_load(sdl) if isinstance(sdl, str) else sdl
    assert sdl_dict["version"] == "2.0"
    assert sdl_dict["services"]["app"]["image"] == "nginx:latest"
    assert sdl_dict["profiles"]["compute"]["app"]["resources"]["cpu"]["units"] == 0.5
    assert sdl_dict["profiles"]["compute"]["app"]["resources"]["memory"]["size"] == "512Mi"
    assert sdl_dict["profiles"]["compute"]["app"]["resources"]["storage"][0]["size"] == "1Gi"
    assert sdl_dict["services"]["app"]["expose"][0]["port"] == 80

@patch('src.akash.akash_manager.client')
def test_deploy_to_akash_success(mock_client, session, mock_wallet, caplog):
    """Test successful deployment to Akash."""
    mock_client.create_deployment.return_value = {"txhash": "test_tx_hash"}
    mock_dseq = str(uuid.uuid4())

    mock_client.broadcast_tx.return_value = {
        "code": 0,
        "txhash": "test_tx_hash",
        "logs": [{"events": [{"attributes": [{"value": mock_dseq}]}]}]
    }

    dseq = deploy_to_akash(
        wallet=mock_wallet,
        wallet_address="akash1testwalletaddress",
        image="nginx:latest",
        cpu=0.5,
        memory="512Mi",
        storage="1Gi",
        ports=["80"],
        db_session=session
    )

    assert dseq == mock_dseq

    deployment = session.query(Deployment).filter_by(deployment_id=mock_dseq).first()
    assert deployment is not None
    assert deployment.image == "nginx:latest"
    assert deployment.status == "pending"
    assert deployment.cpu == 0.5
    assert deployment.memory == "512Mi"
    assert deployment.storage == "1Gi"
    assert deployment.ports == json.dumps(["80"])
    assert f"Deployment created with ID: {mock_dseq}" in caplog.text

@patch('src.akash.akash_manager.client')
def test_deploy_to_akash_failure(mock_client, mock_wallet, caplog):
    """Test deployment failure due to broadcast error."""
    mock_client.create_deployment.return_value = {"txhash": "test_tx_hash"}
    mock_client.broadcast_tx.return_value = {"code": 1, "raw_log": "insufficient funds"}
    
    with pytest.raises(Exception, match="Deployment failed: insufficient funds"):
        deploy_to_akash(mock_wallet, "akash1testwalletaddress", "nginx:latest", 0.5, "512Mi", "1Gi", ["80"])
    assert "Deployment error for akash1testwalletaddress" in caplog.text

@patch('src.akash.akash_manager.client')
def test_get_deployment_status_success(mock_client, session, caplog):
    """Test successful retrieval of deployment status."""
    mock_dseq = str(uuid.uuid4())

    deployment = Deployment(
        deployment_id=mock_dseq,
        wallet_address="akash1testwalletaddress",
        status="pending",
        image="nginx:latest",
        cpu=0.5,
        memory="512Mi",
        storage="1Gi",
        ports=json.dumps(["80"]),
        created_at=datetime.utcnow()
    )
    session.add(deployment)
    session.commit()

    mock_client.query_deployment_status.return_value = {"state": "active"}

    status = get_deployment_status("akash1testwalletaddress", mock_dseq, db_session=session)
    assert status == "active"

    updated_deployment = session.query(Deployment).filter_by(deployment_id=mock_dseq).first()
    assert updated_deployment.status == "active"
    assert f"Deployment {mock_dseq} status updated to active" in caplog.text

@patch('src.akash.akash_manager.client')
def test_get_deployment_status_unauthorized(mock_client, session, caplog):
    """Test deployment status retrieval with unauthorized wallet address."""
    mock_dseq = str(uuid.uuid4())

    deployment = Deployment(
        deployment_id=mock_dseq,
        wallet_address="akash1otherwallet",
        status="pending",
        image="nginx:latest",
        cpu=0.5,
        memory="512Mi",
        storage="1Gi",
        ports=json.dumps(["80"]),
        created_at=datetime.utcnow()
    )
    session.add(deployment)
    session.commit()
    
    status = get_deployment_status("akash1testwalletaddress", mock_dseq)
    assert status == "Not found"
    assert f"Deployment {mock_dseq} not found for wallet: akash1testwalletaddress" in caplog.text

@patch('src.akash.akash_manager.client')
def test_get_deployment_status_not_found(mock_client, session, caplog):
    """Test deployment status retrieval for non-existent deployment."""
    mock_client.query_deployment.side_effect = Exception("Deployment not found")

    status = get_deployment_status("akash1testwalletaddress", "nonexistent_dseq")
    assert status == "Not found"

    assert "Deployment nonexistent_dseq not found for wallet: akash1testwalletaddress" in caplog.text

@patch('src.akash.akash_manager.client')
def test_terminate_deployment_success(mock_client, session, mock_wallet, caplog):
    """Test successful termination of a deployment."""
    mock_dseq = str(uuid.uuid4())

    deployment = Deployment(
        deployment_id=mock_dseq,
        wallet_address="akash1testwalletaddress",
        status="active",
        image="nginx:latest",
        cpu=0.5,
        memory="512Mi",
        storage="1Gi",
        ports=json.dumps(["80"]),
        created_at=datetime.utcnow()
    )
    session.add(deployment)
    session.commit()

    mock_client.close_deployment.return_value = {"txhash": "test_tx_hash"}
    mock_client.broadcast_tx.return_value = {"code": 0}

    result = terminate_deployment("akash1testwalletaddress", mock_dseq, mock_wallet, db_session=session)
    assert result is True

    updated = session.query(Deployment).filter_by(deployment_id=mock_dseq).first()
    assert updated.status == "closed"
    assert f"Deployment {mock_dseq} terminated successfully" in caplog.text

@patch('src.akash.akash_manager.client')
def test_terminate_deployment_unauthorized(mock_client, session, mock_wallet, caplog):
    """Test termination with unauthorized wallet address."""
    mock_dseq = str(uuid.uuid4())

    deployment = Deployment(
        deployment_id=mock_dseq,
        wallet_address="akash1otherwallet",  # belongs to someone else!
        status="active",
        image="nginx:latest",
        cpu=0.5,
        memory="512Mi",
        storage="1Gi",
        ports=json.dumps(["80"]),
        created_at=datetime.utcnow()
    )
    session.add(deployment)
    session.commit()

    result = terminate_deployment("akash1testwalletaddress", mock_dseq, mock_wallet, db_session=session)
    assert result is False

    unchanged = session.query(Deployment).filter_by(deployment_id=mock_dseq).first()
    assert unchanged.status == "active"
    assert f"Deployment {mock_dseq} not found for wallet: akash1testwalletaddress" in caplog.text

@patch('src.akash.akash_manager.client')
def test_terminate_deployment_failure(mock_client, session, mock_wallet, caplog):
    """Test termination failure due to broadcast error."""
    mock_dseq = str(uuid.uuid4())

    deployment = Deployment(
        deployment_id=mock_dseq,
        wallet_address="akash1testwalletaddress",
        status="active",
        image="nginx:latest",
        cpu=0.5,
        memory="512Mi",
        storage="1Gi",
        ports=json.dumps(["80"]),
        created_at=datetime.utcnow()
    )
    session.add(deployment)
    session.commit()

    mock_client.close_deployment.return_value = {"txhash": "test_tx_hash"}
    mock_client.broadcast_tx.return_value = {"code": 1, "raw_log": "termination failed"}

    result = terminate_deployment("akash1testwalletaddress", mock_dseq, mock_wallet, db_session=session)
    assert result is False

    unchanged = session.query(Deployment).filter_by(deployment_id=mock_dseq).first()
    assert unchanged.status == "active"
    assert f"Termination failed for {mock_dseq}: termination failed" in caplog.text

def test_get_all_deployments_pagination(session, caplog):
    """Test pagination for retrieving all deployments."""
    session.query(Deployment).filter_by(wallet_address="akash1testwalletaddress").delete()
    session.commit()

    deployments = [
        Deployment(
            deployment_id=f"test_dseq_{i}",
            wallet_address="akash1testwalletaddress",
            status="active",
            image="nginx:latest",
            cpu=0.5,
            memory="512Mi",
            storage="1Gi",
            ports=json.dumps(["80"]),
            created_at=datetime.utcnow()
        ) for i in range(15)
    ]
    session.add_all(deployments)
    session.commit()

    result = get_all_deployments("akash1testwalletaddress", page=2, per_page=5, db_session=session)

    assert len(result["deployments"]) == 5
    assert result["page"] == 2
    assert result["per_page"] == 5
    assert result["total"] == 15
    assert result["total_pages"] == 3

def test_get_all_deployments_empty(session, caplog):
    """Test retrieval of deployments when none exist."""
    session.query(Deployment).delete()
    session.commit()

    result = get_all_deployments("akash1testwalletaddress", page=1, per_page=10, db_session=session)

    assert len(result["deployments"]) == 0
    assert result["total"] == 0
    assert result["page"] == 1
    assert result["total_pages"] == 0
    assert "No deployments found for akash1testwalletaddress" in caplog.text

def test_invalid_deployment_id(mock_wallet, caplog):
    """Test handling of invalid deployment IDs."""
    with pytest.raises(ValueError, match="Invalid deployment ID format"):
        get_deployment_status("akash1testwalletaddress", "invalid@id")

    with pytest.raises(ValueError, match="Invalid deployment ID format"):
        terminate_deployment("akash1testwalletaddress", "invalid@id", mock_wallet)
