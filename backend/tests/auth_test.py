import pytest
import hashlib
import base64
import logging
import requests
from datetime import datetime, timezone
from unittest.mock import Mock, patch
from ecdsa import SECP256k1, SigningKey
from bech32 import bech32_encode, convertbits
import jwt
from src.auth.auth import Settings, AkashSignatureVerifier, verify_signature, create_session, validate_session, SESSION_DURATION, PUBKEY_CACHE_TTL
from pydantic import ValidationError, AnyHttpUrl
from unittest.mock import MagicMock
from ecdsa.util import sigencode_string_canonize
from base64 import b64encode
from freezegun import freeze_time

# Test setup
@pytest.fixture(autouse=True)
def setup_env():
    """Load environment variables from .env.test."""
    from dotenv import load_dotenv
    load_dotenv('.env.test', override=True, verbose=True)
    yield

@pytest.fixture
def settings_fixture(setup_env):
    """Provide a Settings instance for tests."""
    try:
        return Settings()
    except ValidationError as e:
        pytest.fail(f"Environment variable validation failed: {e}")

@pytest.fixture
def mock_redis(mocker):
    """Mock Redis client."""
    return mocker.patch('redis.Redis', autospec=True)

@pytest.fixture
def mock_requests(mocker):
    """Mock requests.get for HTTP calls."""
    return mocker.patch('requests.get')

@pytest.fixture
def valid_wallet_address():
    """Generate a valid Akash address (bech32 encoded)."""
    hrp = 'akash'
    data = convertbits(bytes([0] * 20), 8, 5, True)
    return bech32_encode(hrp, data)

@pytest.fixture
def valid_key_pair():
    """Generate a valid ECDSA key pair for testing."""
    sk = SigningKey.generate(curve=SECP256k1)
    vk = sk.get_verifying_key()
    return sk, vk

@pytest.fixture
def caplog(caplog):
    """Capture log messages."""
    caplog.set_level(logging.DEBUG)
    return caplog

def test_settings_validation(setup_env):
    """Test Settings validation with correct environment variables from .env.test."""
    try:
        settings = Settings()
        assert settings.database_url == "sqlite:///:memory:"
        assert settings.redis_url == "redis://localhost:6379"
        assert settings.akash_lcd_endpoint == AnyHttpUrl('http://lcd.testnet.akash.network')
        # Avoid asserting sensitive data like akash_mnemonic or secret_key directly
        assert settings.akash_mnemonic is not None  # Check that mnemonic is set
        assert settings.secret_key is not None  # Check that secret_key is derived
    except ValidationError:
        pytest.fail("Settings validation failed with valid .env.test")

def test_akash_signature_verifier_fetch_public_key_success(mock_requests, mock_redis, valid_wallet_address, valid_key_pair, caplog):
    """Test fetching public key successfully with caching."""
    sk, vk = valid_key_pair
    pubkey_bytes = vk.to_string('compressed')
    pubkey_base64 = base64.b64encode(pubkey_bytes).decode('utf-8')

    # Mock API response
    mock_response = Mock()
    mock_response.json.return_value = {'account': {'pub_key': {'key': pubkey_base64}}}
    mock_response.raise_for_status = Mock()
    mock_requests.return_value = mock_response

    # Mock Redis behavior
    mock_redis_instance = Mock()
    mock_redis_instance.get.return_value = None
    mock_redis_instance.setex = Mock()
    mock_redis.from_url.return_value = mock_redis_instance  # Explicitly mock from_url

    # Initialize verifier and fetch public key
    verifier = AkashSignatureVerifier(valid_wallet_address, redis_client=mock_redis_instance)

    pubkey_hex = verifier.fetch_public_key()

    # Verify results
    assert pubkey_hex == pubkey_bytes.hex()
    mock_requests.assert_called_once_with(
        f"http://lcd.testnet.akash.network/cosmos/auth/v1beta1/accounts/{valid_wallet_address}",
        timeout=5
    )
    mock_redis_instance.setex.assert_called_once_with(
        f"pubkey:{valid_wallet_address}", 600, pubkey_bytes.hex()
    )
    assert f"Fetched and cached public key for {valid_wallet_address}" in caplog.text

def test_akash_signature_verifier_fetch_public_key_cached(mock_redis, valid_wallet_address, valid_key_pair, caplog):
    """Test fetching cached public key."""
    sk, vk = valid_key_pair
    pubkey_bytes = vk.to_string('compressed')
    pubkey_hex = pubkey_bytes.hex()

    # Setup deep Redis mock
    mock_redis_instance = MagicMock()
    mock_redis_instance.get.return_value = pubkey_hex
    mock_redis.return_value = mock_redis_instance

    # Instantiate verifier and run
    verifier = AkashSignatureVerifier(valid_wallet_address, redis_client=mock_redis_instance)
    fetched_hex = verifier.fetch_public_key()

    # Assertions
    assert fetched_hex == pubkey_hex
    mock_redis_instance.get.assert_called_once_with(f"pubkey:{valid_wallet_address}")
    assert f"Retrieved cached public key for {valid_wallet_address}" in caplog.text

@patch("src.auth.auth.redis.Redis.from_url")
@patch("src.auth.auth.requests.get")
def test_akash_signature_verifier_fetch_public_key_failure(mock_get, mock_redis_from_url, valid_wallet_address, caplog):
    """Test public key fetch failure due to RequestException."""
    # Mock request failure
    mock_get.side_effect = requests.RequestException("Connection error")

    # Mock Redis behavior
    mock_redis_instance = MagicMock()
    mock_redis_instance.get.return_value = None
    mock_redis_from_url.return_value = mock_redis_instance

    # Run verification
    verifier = AkashSignatureVerifier(valid_wallet_address)
    result = verifier.fetch_public_key()

    # Validate
    assert result is None
    mock_get.assert_called_once()
    mock_redis_instance.get.assert_called_once_with(f"pubkey:{valid_wallet_address}")
    assert f"Failed to fetch public key for {valid_wallet_address}" in caplog.text
    
@patch("src.auth.auth.redis.Redis.from_url") 
@patch("src.auth.auth.requests.get")
def test_akash_signature_verifier_fetch_public_key_missing_pubkey(mock_get, mock_redis_from_url, valid_wallet_address, caplog):
    """Test public key fetch with missing pubkey in response."""
    # Simulate valid response but empty pub_key
    mock_response = Mock()
    mock_response.json.return_value = {'account': {'pub_key': {}}}
    mock_response.raise_for_status = Mock()
    mock_get.return_value = mock_response

    # Patch Redis with None for get()
    mock_redis_instance = MagicMock()
    mock_redis_instance.get.return_value = None
    mock_redis_from_url.return_value = mock_redis_instance

    # Run
    verifier = AkashSignatureVerifier(valid_wallet_address)
    result = verifier.fetch_public_key()

    # Assertions
    assert result is None
    mock_get.assert_called_once()
    mock_redis_instance.get.assert_called_once_with(f"pubkey:{valid_wallet_address}")
    assert f"Public key not found for address {valid_wallet_address}" in caplog.text

@patch("src.auth.auth.redis.Redis.from_url")
@patch("src.auth.auth.requests.get")
def test_akash_signature_verifier_verify_signature_success(mock_get, mock_redis_from_url):
    """Test successful signature verification using compressed pubkey + address match."""
    # Step 1: Generate valid keypair
    sk = SigningKey.generate(curve=SECP256k1)
    vk = sk.verifying_key

    # Step 2: Generate valid Akash address (bech32 from public key hash)
    pubkey_uncompressed = vk.to_string()
    pubkey_hash = hashlib.new('ripemd160', hashlib.sha256(pubkey_uncompressed).digest()).digest()
    bech32_data = convertbits(pubkey_hash, 8, 5)
    wallet_address = bech32_encode('akash', bech32_data)

    # Step 3: Compress the pubkey for mocking LCD output
    pubkey_bytes = vk.to_string("compressed")
    pubkey_base64 = b64encode(pubkey_bytes).decode("utf-8")

    # Step 4: Mock LCD response
    mock_response = Mock()
    mock_response.json.return_value = {
        "account": {"pub_key": {"key": pubkey_base64}}
    }
    mock_response.raise_for_status = Mock()
    mock_get.return_value = mock_response

    # Step 5: Mock Redis
    mock_redis_instance = MagicMock()
    mock_redis_instance.get.return_value = None
    mock_redis_instance.setex = Mock()
    mock_redis_from_url.return_value = mock_redis_instance

    # Step 6: Sign the nonce
    nonce = "test-nonce-123"
    msg_hash = hashlib.sha256(nonce.encode("utf-8")).digest()
    signature = sk.sign(msg_hash, sigencode=sigencode_string_canonize).hex()

    # Step 7: Run verification
    verifier = AkashSignatureVerifier(wallet_address)
    result = verifier.verify_signature(nonce, signature)

    # Step 8: Assertions
    assert result is True
    mock_get.assert_called_once()
    mock_redis_instance.setex.assert_called_once_with(f"pubkey:{wallet_address}", PUBKEY_CACHE_TTL, pubkey_bytes.hex())

@patch("src.auth.auth.redis.Redis.from_url")
@patch("src.auth.auth.requests.get")
def test_akash_signature_verifier_verify_signature_invalid_signature(mock_get, mock_redis_from_url):
    """Test signature verification with clearly invalid signature (64 zero bytes)."""
    # Step 1: Generate valid keypair
    sk = SigningKey.generate(curve=SECP256k1)
    vk = sk.verifying_key

    # Step 2: Generate valid Akash address from pubkey
    pubkey_uncompressed = vk.to_string()
    pubkey_hash = hashlib.new("ripemd160", hashlib.sha256(pubkey_uncompressed).digest()).digest()
    bech32_data = convertbits(pubkey_hash, 8, 5)
    wallet_address = bech32_encode("akash", bech32_data)

    # Step 3: Prepare LCD mock response with compressed pubkey
    pubkey_bytes = vk.to_string("compressed")
    pubkey_base64 = b64encode(pubkey_bytes).decode("utf-8")

    mock_response = Mock()
    mock_response.json.return_value = {
        "account": {"pub_key": {"key": pubkey_base64}}
    }
    mock_response.raise_for_status = Mock()
    mock_get.return_value = mock_response

    # Step 4: Prepare Redis mocks
    mock_redis_instance = MagicMock()
    mock_redis_instance.get.return_value = None
    mock_redis_instance.setex = Mock()
    mock_redis_from_url.return_value = mock_redis_instance

    # Step 5: Use clearly invalid 64-byte signature
    invalid_signature = bytes([0] * 64).hex()
    nonce = "test-nonce-123"

    # Step 6: Run verifier
    verifier = AkashSignatureVerifier(wallet_address)
    result = verifier.verify_signature(nonce, invalid_signature)

    # Step 7: Assertions
    assert result is False
    mock_get.assert_called_once()
    mock_redis_instance.setex.assert_called_once_with(f"pubkey:{wallet_address}", 600, pubkey_bytes.hex())

def test_akash_signature_verifier_verify_signature_invalid_address(caplog):
    """Test signature verification with invalid address."""
    verifier = AkashSignatureVerifier("invalid_address")
    result = verifier.verify_signature("nonce", "A" * 128)
    assert result is False
    assert f"Invalid Akash address: invalid_address" in caplog.text

@patch("src.auth.auth.redis.Redis.from_url")
@patch("src.auth.auth.requests.get")
def test_akash_signature_verifier_verify_signature_invalid_pubkey_length(mock_get, mock_redis_from_url, valid_wallet_address, caplog):
    """Test signature verification fails with malformed (32-byte) public key."""
    # Step 1: Prepare mocked LCD response with 32-byte pubkey (invalid)
    invalid_pubkey = base64.b64encode(b"A" * 32).decode("utf-8")
    mock_response = Mock()
    mock_response.json.return_value = {"account": {"pub_key": {"key": invalid_pubkey}}}
    mock_response.raise_for_status = Mock()
    mock_get.return_value = mock_response

    # Step 2: Mock Redis to simulate a cache miss
    mock_redis_instance = MagicMock()
    mock_redis_instance.get.return_value = None
    mock_redis_instance.setex = Mock()
    mock_redis_from_url.return_value = mock_redis_instance

    # Step 3: Use a fake but structurally valid signature
    fake_signature = "A" * 128  # 64-byte hex string

    # Step 4: Invoke verifier
    verifier = AkashSignatureVerifier(valid_wallet_address)
    result = verifier.verify_signature("nonce", fake_signature)

    # Step 5: Assert logic flow
    assert result is False
    assert f"Invalid public key length for {valid_wallet_address}" in caplog.text
    mock_redis_instance.setex.assert_called_once()
    mock_get.assert_called_once()
    
@patch("src.auth.auth.redis.Redis.from_url")
@patch("src.auth.auth.requests.get")
def test_akash_signature_verifier_verify_signature_invalid_signature_length(mock_get, mock_redis_from_url, caplog):
    """Test signature verification fails with malformed signature length (not 64 bytes)."""
    # Step 1: Generate proper key pair and Akash address
    sk = SigningKey.generate(curve=SECP256k1)
    vk = sk.verifying_key
    pubkey_uncompressed = vk.to_string()
    pubkey_hash = hashlib.new("ripemd160", hashlib.sha256(pubkey_uncompressed).digest()).digest()
    bech32_data = convertbits(pubkey_hash, 8, 5)
    wallet_address = bech32_encode("akash", bech32_data)

    # Step 2: LCD response with valid 33-byte compressed pubkey
    pubkey_compressed = vk.to_string("compressed")
    pubkey_base64 = b64encode(pubkey_compressed).decode("utf-8")

    mock_response = Mock()
    mock_response.json.return_value = {"account": {"pub_key": {"key": pubkey_base64}}}
    mock_response.raise_for_status = Mock()
    mock_get.return_value = mock_response

    # Step 3: Mock Redis
    mock_redis_instance = MagicMock()
    mock_redis_instance.get.return_value = None
    mock_redis_instance.setex = Mock()
    mock_redis_from_url.return_value = mock_redis_instance

    # Step 4: Use invalid signature length (e.g., 63 bytes instead of 64)
    invalid_signature = "ab" * 63  # 126 hex chars → 63 bytes
    verifier = AkashSignatureVerifier(wallet_address)
    result = verifier.verify_signature("some-nonce", invalid_signature)

    # Step 5: Assertions
    assert result is False
    assert f"Invalid signature length for {wallet_address}" in caplog.text

    
@patch("src.auth.auth.redis.Redis.from_url")
@patch("src.auth.auth.requests.get")
def test_akash_signature_verifier_verify_signature_pubkey_mismatch(mock_get, mock_redis_from_url, caplog):
    """Test that a valid signature from a mismatched key fails address pubkey validation."""

    # Generate signing key (sk) and unrelated verifying key (vk2)
    sk = SigningKey.generate(curve=SECP256k1)
    vk_signing = sk.verifying_key
    vk2 = SigningKey.generate(curve=SECP256k1).verifying_key

    # Create address based on vk2 (mismatched pubkey used to derive address)
    pubkey_bytes_vk2_uncompressed = vk2.to_string()
    pubkey_hash = hashlib.new("ripemd160", hashlib.sha256(pubkey_bytes_vk2_uncompressed).digest()).digest()
    bech32_data = convertbits(pubkey_hash, 8, 5)
    wallet_address = bech32_encode("akash", bech32_data)

    # Set up the LCD mock to return vk_signing (the correct key for the signature)
    pubkey_compressed = vk_signing.to_string("compressed")
    pubkey_base64 = base64.b64encode(pubkey_compressed).decode("utf-8")
    mock_response = Mock()
    mock_response.json.return_value = {"account": {"pub_key": {"key": pubkey_base64}}}
    mock_response.raise_for_status = Mock()
    mock_get.return_value = mock_response

    # Mock Redis
    mock_redis_instance = MagicMock()
    mock_redis_instance.get.return_value = None
    mock_redis_instance.setex = Mock()
    mock_redis_from_url.return_value = mock_redis_instance

    # Sign nonce
    nonce = "test-nonce-123"
    msg_hash = hashlib.sha256(nonce.encode("utf-8")).digest()
    signature = sk.sign(msg_hash, sigencode=sigencode_string_canonize).hex()

    # Attempt verification
    verifier = AkashSignatureVerifier(wallet_address)
    result = verifier.verify_signature(nonce, signature)

    # Assert logic failure is due to pubkey mismatch, not signature format error
    assert result is False
    assert f"Public key mismatch for address {wallet_address}" in caplog.text

@patch("src.auth.auth.redis.Redis.from_url")
@patch("src.auth.auth.requests.get")
def test_verify_signature_wrapper(mock_get, mock_redis_from_url, caplog):
    """Test verify_signature wrapper function with a valid signature."""

    # Generate valid key pair
    sk = SigningKey.generate(curve=SECP256k1)
    vk = sk.verifying_key

    # Compute valid Akash address
    pubkey_uncompressed = vk.to_string()
    pubkey_hash = hashlib.new("ripemd160", hashlib.sha256(pubkey_uncompressed).digest()).digest()
    bech32_bits = convertbits(pubkey_hash, 8, 5)
    wallet_address = bech32_encode("akash", bech32_bits)

    # Encode compressed pubkey for mock LCD response
    pubkey_compressed = vk.to_string("compressed")
    pubkey_base64 = base64.b64encode(pubkey_compressed).decode("utf-8")

    # Mock LCD response
    mock_response = Mock()
    mock_response.json.return_value = {"account": {"pub_key": {"key": pubkey_base64}}}
    mock_response.raise_for_status = Mock()
    mock_get.return_value = mock_response

    # Mock Redis
    mock_redis_instance = MagicMock()
    mock_redis_instance.get.return_value = None
    mock_redis_instance.setex = Mock()
    mock_redis_from_url.return_value = mock_redis_instance

    # Sign nonce
    nonce = "test-nonce-123"
    msg_hash = hashlib.sha256(nonce.encode("utf-8")).digest()
    signature = sk.sign(msg_hash, sigencode=sigencode_string_canonize).hex()

    # Verify using wrapper
    result = verify_signature(wallet_address, nonce, signature)

    # Assertions
    assert result is True
    assert f"Signature verification successful for {wallet_address}" in caplog.text

def test_create_session_success(valid_wallet_address, settings_fixture, caplog):
    """Test successful JWT creation."""
    with patch('src.auth.auth.settings', settings_fixture):
        with patch('src.auth.auth.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2025, 7, 2, tzinfo=timezone.utc)
            token = create_session(valid_wallet_address)
    
    payload = jwt.decode(
        token,
        settings_fixture.secret_key,
        algorithms=['HS256'],
        options={"verify_exp": False}
    )
    assert payload['wallet_address'] == valid_wallet_address
    assert 'exp' in payload
    assert 'iat' in payload
    assert payload['exp'] == int((datetime(2025, 7, 2, tzinfo=timezone.utc) + SESSION_DURATION).timestamp())
    assert f"Created new JWT for {valid_wallet_address}" in caplog.text

def test_create_session_error(valid_wallet_address, settings_fixture, caplog):
    """Test JWT creation error."""
    with patch('src.auth.auth.settings', settings_fixture):
        with patch('jwt.encode', side_effect=Exception("Encoding error")):
            with pytest.raises(Exception, match="Encoding error"):
                create_session(valid_wallet_address)
    assert f"Error creating JWT for {valid_wallet_address}" in caplog.text

@freeze_time("2025-07-02T00:00:00Z")  # ISO format, UTC time
def test_validate_session_success(valid_wallet_address, settings_fixture, caplog):
    """Test successful JWT validation."""
    with patch("src.auth.auth.settings", settings_fixture):
        token = create_session(valid_wallet_address)
        wallet_addr, error, status = validate_session(token)

        assert wallet_addr == valid_wallet_address
        assert error is None
        assert status == 200
        assert f"Valid JWT for {valid_wallet_address}" in caplog.text

def test_validate_session_missing_token(caplog):
    """Test JWT validation with missing token."""
    wallet_addr, error, status = validate_session("")
    
    assert wallet_addr is None
    assert error == {"msg": "JWT required"}
    assert status == 401
    assert "Missing JWT in validate_session" in caplog.text

def test_validate_session_expired_token(valid_wallet_address, settings_fixture, caplog):
    """Test JWT validation with expired token."""
    with patch('src.auth.auth.settings', settings_fixture):
        with patch('src.auth.auth.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2025, 7, 2, tzinfo=timezone.utc)
            token = create_session(valid_wallet_address)
            # Simulate expiration
            mock_datetime.now.return_value = datetime(2025, 7, 3, tzinfo=timezone.utc)
    
        wallet_addr, error, status = validate_session(token)
    
        assert wallet_addr is None
        assert error == {"msg": "Expired JWT"}
        assert status == 401
        assert "Expired JWT" in caplog.text

def test_validate_session_invalid_token(caplog):
    """Test JWT validation with invalid token."""
    wallet_addr, error, status = validate_session("invalid.token.here")
    
    assert wallet_addr is None
    assert error == {"msg": "Invalid JWT"}
    assert status == 401
    assert "Invalid JWT format" in caplog.text

def test_validate_session_missing_wallet_address(valid_wallet_address, settings_fixture, caplog):
    """Test JWT validation with missing wallet_address in payload."""
    with patch('src.auth.auth.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2025, 7, 2, tzinfo=timezone.utc)
        payload = {
            'exp': datetime.now(timezone.utc) + SESSION_DURATION,
            'iat': datetime.now(timezone.utc)
        }
        token = jwt.encode(payload, settings_fixture.secret_key, algorithm='HS256')
    
    wallet_addr, error, status = validate_session(token)
    
    assert wallet_addr is None
    assert error == {"msg": "Invalid JWT"}
    assert status == 401
    assert "Invalid JWT: missing wallet_address" in caplog.text

def test_validate_session_error(valid_wallet_address, settings_fixture, caplog):
    """Test JWT validation with unexpected error."""
    with patch('src.auth.auth.settings', settings_fixture):
        with patch('jwt.decode', side_effect=Exception("Decoding error")):
            with patch('src.auth.auth.datetime') as mock_datetime:
                mock_datetime.now.return_value = datetime(2025, 7, 2, tzinfo=timezone.utc)
                token = create_session(valid_wallet_address)
        
            wallet_addr, error, status = validate_session(token)
        
            assert wallet_addr is None
            assert error == {"msg": "Internal server error"}
            assert status == 500
            assert "Error validating JWT" in caplog.text
            