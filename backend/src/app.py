import os
import re
import logging
from logging import handlers
from dotenv import find_dotenv, load_dotenv
from flask import Flask, jsonify, request, render_template
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_cors import CORS
import speech_recognition as sr
from cosmpy.aerial.wallet import LocalWallet
from src.auth.auth import verify_signature, create_session, validate_session, settings
from src.akash.akash_manager import deploy_to_akash, get_deployment_status, terminate_deployment, get_all_deployments
from bech32 import bech32_decode
from src.voice.voice_parser import parse_voice_command

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        handlers.RotatingFileHandler('app.log', maxBytes=10_000_000, backupCount=5),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
dotenv_path = find_dotenv(usecwd=True)
load_dotenv(dotenv_path, override=True, verbose=True)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = settings.secret_key

# Configure CORS
allowed_origin = os.getenv('ALLOWED_ORIGIN', 'https://your-trusted-domain.com')
CORS(app, resources={r"/*": {"origins": [allowed_origin]}})

# Configure rate limiter
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri=settings.redis_url
)
recognizer = sr.Recognizer()

# Initialize Akash wallet
try:
    wallet = LocalWallet.from_mnemonic(os.getenv("AKASH_MNEMONIC"))
    logger.info(f"Akash Wallet Initialized: {str(wallet.address())}")
except Exception as e:
    logger.error(f"Failed to initialize Akash wallet: {str(e)}")
    raise

# Input validation patterns
WALLET_ADDRESS_PATTERN = re.compile(r'^akash1[0-9a-z]{38}$')
NONCE_PATTERN = re.compile(r'^[a-zA-Z0-9]{1,100}$')
SIGNATURE_PATTERN = re.compile(r'^[0-9a-fA-F]{128}$')

def validate_wallet_address(wallet_address: str) -> bool:
    """Validate Akash wallet address format and bech32 encoding."""
    if not WALLET_ADDRESS_PATTERN.match(wallet_address):
        logger.warning(f"Invalid wallet address format: {wallet_address}")
        return False
    hrp, data = bech32_decode(wallet_address)
    if hrp != 'akash' or not data:
        logger.warning(f"Invalid bech32 encoding for wallet address: {wallet_address}")
        return False
    return True

def validate_nonce(nonce: str) -> bool:
    """Validate nonce format."""
    if not NONCE_PATTERN.match(nonce):
        logger.warning(f"Invalid nonce format: {nonce}")
        return False
    return True

def validate_signature(signature: str) -> bool:
    """Validate signature format."""
    if not SIGNATURE_PATTERN.match(signature):
        logger.warning(f"Invalid signature format: {signature}")
        return False
    return True

@app.route('/')
def home():
    """Root endpoint."""
    logger.debug("Accessed home endpoint")
    return jsonify({"OK": True}), 200

@app.route('/create_session', methods=['POST'])
@limiter.limit("10 per minute")
def create_session_endpoint():
    """Create a new JWT for a verified wallet address."""
    try:
        data = request.get_json()
        wallet_address = data.get('wallet_address')
        nonce = data.get('nonce')
        signature = data.get('signature')

        if not all([wallet_address, nonce, signature]):
            logger.warning("Missing required fields in create_session")
            return jsonify({"msg": "Missing wallet_address, nonce, or signature"}), 400
        
        if not validate_wallet_address(wallet_address):
            return jsonify({"msg": "Invalid wallet address format"}), 400
        
        if not validate_nonce(nonce):
            return jsonify({"msg": "Invalid nonce format"}), 400
        
        if not validate_signature(signature):
            return jsonify({"msg": "Invalid signature format"}), 400

        if not verify_signature(wallet_address, nonce, signature):
            logger.warning(f"Signature verification failed for {wallet_address}")
            return jsonify({"msg": "Invalid signature"}), 401

        token = create_session(wallet_address)
        logger.info(f"JWT created for {wallet_address}")
        return jsonify({"token": token}), 200
    except Exception as e:
        logger.error(f"Error in create_session: {str(e)}")
        return jsonify({"msg": "Internal server error"}), 500

@app.route('/refresh_token', methods=['POST'])
@limiter.limit("10 per minute")
def refresh_token():
    """Refresh an existing JWT to extend its expiration."""
    try:
        token = request.headers.get('X-Token') or request.args.get('token')
        if not token:
            logger.warning("Missing JWT in refresh_token")
            return jsonify({"msg": "JWT required"}), 401

        wallet_address, error, status = validate_session(token)
        if error:
            logger.warning(f"JWT validation failed in refresh_token: {error['msg']}")
            return jsonify(error), status

        new_token = create_session(wallet_address)
        logger.info(f"Refreshed JWT for {wallet_address}")
        return jsonify({"token": new_token}), 200
    except Exception as e:
        logger.error(f"Error in refresh_token: {str(e)}")
        return jsonify({"msg": "Internal server error"}), 500

@app.route('/validate_session', methods=['GET'])
@limiter.limit("20 per minute")
def validate_session_endpoint():
    """Validate an existing JWT."""
    token = request.headers.get('X-Token') or request.args.get('token')
    if not token:
        logger.warning("Missing JWT in validate_session")
        return jsonify({"msg": "JWT required"}), 401

    wallet_address, error, status = validate_session(token)
    if error:
        logger.warning(f"JWT validation failed: {error['msg']}")
        return jsonify(error), status
    logger.debug(f"JWT validated for {wallet_address}")
    return jsonify({"wallet_address": wallet_address}), 200

@app.route('/login', methods=['POST'])
@limiter.limit("10 per minute")
def login():
    """Handle user login with wallet address and signature."""
    try:
        data = request.form
        wallet_address = data.get('wallet_address')
        nonce = data.get('nonce')
        signature = data.get('signature')

        if not all([wallet_address, nonce, signature]):
            logger.warning("Missing required fields in login")
            return jsonify({"msg": "Missing wallet_address, nonce, or signature"}), 400
        
        if not validate_wallet_address(wallet_address):
            return jsonify({"msg": "Invalid wallet address format"}), 400
        
        if not validate_nonce(nonce):
            return jsonify({"msg": "Invalid nonce format"}), 400
        
        if not validate_signature(signature):
            return jsonify({"msg": "Invalid signature format"}), 400

        if verify_signature(wallet_address, nonce, signature):
            token = create_session(wallet_address)
            logger.info(f"Login successful for {wallet_address}")
            return jsonify({"token": token}), 200
        logger.warning(f"Signature verification failed for {wallet_address}")
        return jsonify({"msg": "Signature verification failed"}), 401
    except Exception as e:
        logger.error(f"Error in login: {str(e)}")
        return jsonify({"msg": "Internal server error"}), 500

@app.route('/voice', methods=['POST'])
@limiter.limit("10 per minute")
def voice_command():
    """Handle voice commands for deployment actions."""
    token = request.headers.get('X-Token') or request.args.get('token')
    if not token:
        logger.warning("Missing JWT in voice_command")
        return jsonify({"msg": "JWT required"}), 401

    wallet_address, error, status = validate_session(token)
    if error:
        logger.warning(f"JWT validation failed: {error['msg']}")
        return jsonify(error), status

    try:
        with sr.Microphone() as source:
            logger.debug("Listening for voice command...")
            audio = recognizer.listen(source, timeout=5)

        command, raw_text = parse_voice_command(audio)
        response = {"raw_text": raw_text}

        if command["action"] == "deploy" and command["target"] == "deployment":
            if not all([command.get("image"), command.get("cpu"), command.get("memory"), command.get("storage"), command.get("ports")]):
                logger.warning("Missing deployment parameters")
                return jsonify({"msg": "Missing deployment parameters"}), 400
            
            deployment_id = deploy_to_akash(
                wallet,
                wallet_address,
                command["image"],
                float(command["cpu"]),
                command["memory"],
                command["storage"],
                command["ports"]
            )
            response["result"] = f"Deployed {command['image']} with ID: {deployment_id}"
        elif command["action"] == "status" and command["target"] == "deployment":
            deployment_id = command.get("id") or request.args.get("id")
            if not deployment_id or not re.match(r'^[a-zA-Z0-9_-]{1,50}$', deployment_id):
                logger.warning("Invalid or missing deployment ID")
                response["result"] = "Please provide a valid deployment ID"
            else:
                status = get_deployment_status(wallet_address, deployment_id)
                response["result"] = f"Status: {status}"
        elif command["action"] == "terminate" and command["target"] == "deployment":
            deployment_id = command.get("id") or request.args.get("id")
            if not deployment_id or not re.match(r'^[a-zA-Z0-9_-]{1,50}$', deployment_id):
                logger.warning("Invalid or missing deployment ID")
                response["result"] = "Please provide a valid deployment ID"
            elif terminate_deployment(wallet_address, deployment_id):
                response["result"] = "Deployment terminated"
            else:
                response["result"] = "Termination failed or ID not found"
        else:
            response["result"] = "Command not recognized"

        logger.info(f"Processed voice command for {wallet_address}: {raw_text}")
        return jsonify(response), 200
    except sr.WaitTimeoutError:
        logger.warning("No audio input received within timeout")
        return jsonify({"msg": "No audio input received within timeout"}), 400
    except Exception as e:
        logger.error(f"Error processing voice command for {wallet_address}: {str(e)}")
        return jsonify({"msg": f"Error processing voice command: {str(e)}"}), 500

@app.route('/status/<deployment_id>')
@limiter.limit("20 per minute")
def status(deployment_id):
    """Get the status of a specific deployment."""
    if not re.match(r'^[a-zA-Z0-9_-]{1,50}$', deployment_id):
        logger.warning(f"Invalid deployment ID format: {deployment_id}")
        return jsonify({"msg": "Invalid deployment ID format"}), 400

    token = request.headers.get('X-Token') or request.args.get('token')
    if not token:
        logger.warning("Missing JWT in status")
        return jsonify({"msg": "JWT required"}), 401

    wallet_address, error, status_code = validate_session(token)
    if error:
        logger.warning(f"JWT validation failed: {error['msg']}")
        return jsonify(error), status_code
    status_data = get_deployment_status(wallet_address, deployment_id)
    logger.debug(f"Retrieved status for deployment {deployment_id}: {status_data}")
    return render_template('status.html', deployment_id=deployment_id, status=status_data)

@app.route('/deployments')
@limiter.limit("20 per minute")
def deployments():
    """Get a paginated list of deployments for the authenticated wallet."""
    token = request.headers.get('X-Token') or request.args.get('token')
    if not token:
        logger.warning("Missing JWT in deployments")
        return jsonify({"msg": "JWT required"}), 401

    wallet_address, error, status = validate_session(token)
    if error:
        logger.warning(f"JWT validation failed: {error['msg']}")
        return jsonify(error), status

    page = request.args.get('page', default=1, type=int)
    per_page = request.args.get('per_page', default=10, type=int)

    if page < 1:
        page = 1
    if per_page < 1 or per_page > 100:
        per_page = 10

    paginated_deployments = get_all_deployments(wallet_address, page, per_page)
    logger.debug(f"Retrieved deployments for {wallet_address}: page {page}, per_page {per_page}")
    return jsonify(paginated_deployments), 200

if __name__ == "__main__":
    port = int(settings.app_port)
    environment = os.getenv('ENVIRONMENT', os.getenv('FLASK_ENV', 'production'))
    ssl_context = 'adhoc' if environment != 'development' else None
    app.run(debug=True, host="0.0.0.0", port=port, ssl_context=ssl_context)