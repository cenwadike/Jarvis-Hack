import redis
import logging
from logging import handlers
import hashlib
import base64
import requests
from datetime import datetime, timedelta, timezone
from typing import Tuple, Optional, Dict
from ecdsa import SECP256k1, VerifyingKey, BadSignatureError
from bech32 import bech32_decode, convertbits
from pydantic_settings import BaseSettings
from pydantic import AnyHttpUrl, ValidationError
import jwt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        handlers.RotatingFileHandler('auth.log', maxBytes=10_000_000, backupCount=5),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Environment variable validation
class Settings(BaseSettings):
    akash_lcd_endpoint: AnyHttpUrl
    akash_mnemonic: str
    akash_keyring_backend: str
    akash_net: AnyHttpUrl
    akash_chain_id: str
    akash_node: AnyHttpUrl
    database_url: str
    redis_url: str
    app_port: str
    environment: str

    @property
    def secret_key(self) -> str:
        """Generate a secret key by hashing the akash_mnemonic."""
        return hashlib.sha256(self.akash_mnemonic.encode('utf-8')).hexdigest()

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

try:
    settings = Settings()
except ValidationError as e:
    logger.error(f"Environment variable validation failed: {e}")
    raise

# Constants
SIGNATURE_LENGTH = 64
PUBKEY_LENGTH = 33
PUBKEY_CACHE_TTL = 600  # 10 minutes in seconds
SESSION_DURATION = timedelta(hours=1)

class SignatureVerifier:
    """Base class for signature verification across different blockchains."""
    def __init__(self, address: str):
        self.address = address

    def fetch_public_key(self) -> Optional[str]:
        """Fetch the public key associated with the address."""
        raise NotImplementedError

    def verify_signature(self, message: str, signature: str) -> bool:
        """Verify a signature for a given message."""
        raise NotImplementedError

class AkashSignatureVerifier(SignatureVerifier):
    """Signature verifier for Akash blockchain."""
    def __init__(self, address: str, redis_client=None):
        super().__init__(address)
        self.redis_client = redis_client or redis.Redis.from_url(settings.redis_url, decode_responses=True)

    def fetch_public_key(self) -> Optional[str]:
        """Fetch the public key for an Akash address from an Akash LCD endpoint."""
        cache_key = f"pubkey:{self.address}"
        cached_pubkey = self.redis_client.get(cache_key)
        if cached_pubkey:
            logger.debug(f"Retrieved cached public key for {self.address}")
            return cached_pubkey

        try:
            # Strip trailing slash from akash_lcd_endpoint to avoid double slashes
            base_endpoint = str(settings.akash_lcd_endpoint).rstrip('/')
            endpoint = f"{base_endpoint}/cosmos/auth/v1beta1/accounts/{self.address}"
            response = requests.get(endpoint, timeout=5)
            response.raise_for_status()
            account_data = response.json()
            pubkey_base64 = account_data['account'].get('pub_key', {}).get('key')
            if not pubkey_base64:
                logger.warning(f"Public key not found for address {self.address}")
                return None
            pubkey_bytes = base64.b64decode(pubkey_base64)
            pubkey_hex = pubkey_bytes.hex()
            self.redis_client.setex(cache_key, PUBKEY_CACHE_TTL, pubkey_hex)
            logger.info(f"Fetched and cached public key for {self.address}")
            return pubkey_hex
        except requests.RequestException as e:
            logger.error(f"Failed to fetch public key for {self.address}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error processing public key for {self.address}: {e}")
            return None

    def verify_signature(self, nonce: str, signature: str) -> bool:
        """Verify an Akash signature using the public key."""
        try:
            hrp, data = bech32_decode(self.address)
            if hrp != 'akash' or not data:
                logger.warning(f"Invalid Akash address: {self.address}")
                return False
            account_hash = bytes(convertbits(data, 5, 8, False))
            pubkey_hex = self.fetch_public_key()
            if not pubkey_hex:
                logger.warning(f"No public key for {self.address}")
                return False
            pubkey_bytes = bytes.fromhex(pubkey_hex)
            if len(pubkey_bytes) != PUBKEY_LENGTH:
                logger.warning(f"Invalid public key length for {self.address}: {len(pubkey_bytes)} bytes")
                return False
            sig_bytes = bytes.fromhex(signature)
            logger.debug(f"Signature length: {len(sig_bytes)} bytes")
            if len(sig_bytes) != SIGNATURE_LENGTH:
                logger.warning(f"Invalid signature length for {self.address}: {len(sig_bytes)} bytes")
                return False
            message_hash = hashlib.sha256(nonce.encode('utf-8')).digest()
            vk = VerifyingKey.from_string(pubkey_bytes, curve=SECP256k1)
            pubkey_point = vk.pubkey.point
            vk_uncompressed = VerifyingKey.from_public_point(pubkey_point, curve=SECP256k1)
            is_valid = vk_uncompressed.verify(sig_bytes, message_hash)
            pubkey_hash = hashlib.new('ripemd160', hashlib.sha256(vk_uncompressed.to_string()).digest()).digest()
            logger.debug(f"Pubkey hash: {pubkey_hash.hex()}, Account hash: {account_hash.hex()}")  # Debug
            if pubkey_hash != account_hash:
                logger.warning(f"Public key mismatch for address {self.address}")
                return False        
            logger.info(f"Signature verification {'successful' if is_valid else 'failed'} for {self.address}")
            return is_valid
        except BadSignatureError as e:
            logger.warning(f"Invalid signature format for {self.address}: {e}")
            return False
        except Exception as e:
            logger.error(f"Signature verification failed for {self.address}: {e}")
            return False
        
def verify_signature(wallet_address: str, nonce: str, signature: str) -> bool:
    """Verify a signature for a given wallet address and nonce."""
    verifier = AkashSignatureVerifier(wallet_address)
    return verifier.verify_signature(nonce, signature)

def create_session(wallet_address: str) -> str:
    """Create a new JWT for a verified wallet address."""
    try:
        payload = {
            'wallet_address': wallet_address,
            'exp': datetime.now(timezone.utc) + SESSION_DURATION,
            'iat': datetime.now(timezone.utc)
        }
        token = jwt.encode(payload, settings.secret_key, algorithm='HS256')
        logger.info(f"Created new JWT for {wallet_address}")
        return token
    except Exception as e:
        logger.error(f"Error creating JWT for {wallet_address}: {e}")
        raise

def validate_session(token: str, now: Optional[datetime] = None) -> Tuple[Optional[str], Optional[Dict[str, str]], int]:
    """Validate a JWT and return (wallet_address, error, status)."""
    if not token:
        logger.warning("Missing JWT in validate_session")
        return None, {"msg": "JWT required"}, 401

    try:
        payload = jwt.decode(
            token,
            settings.secret_key,
            algorithms=['HS256'],
            options={"verify_exp": True},
            leeway=0,
            current_time=int(now.timestamp()) if now else None
        )
        wallet_address = payload.get('wallet_address')
        if not wallet_address:
            logger.warning("Invalid JWT: missing wallet_address")
            return None, {"msg": "Invalid JWT"}, 401
        logger.debug(f"Valid JWT for {wallet_address}")
        return wallet_address, None, 200
    except jwt.ExpiredSignatureError:
        logger.warning("Expired JWT")
        return None, {"msg": "Expired JWT"}, 401
    except jwt.InvalidTokenError:
        logger.warning("Invalid JWT format")
        return None, {"msg": "Invalid JWT"}, 401
    except Exception as e:
        logger.error(f"Error validating JWT: {e}")
        return None, {"msg": "Internal server error"}, 500
    