# Jarvis-Hack
 
## Project Overview

Jarvis, the Voice AI Akash Cloud Manager, is an innovative decentralized application that 
enables users to deploy, monitor, and manage cloud resources on the Akash Network using voice 
commands. Designed to enhance accessibility and streamline interaction with decentralized 
infrastructure, Jarvis leverages natural language processing (NLP) to interpret user instructions, 
interfacing with the Akash blockchain via cosmpy and the Akash CLI. Deployment data is stored 
persistently in a PostgreSQL database, and ATOM (the Cosmos Hub token) is utilized for payments. 
This project reimagines cloud management by combining voice-driven AI with the Cosmos ecosystem’s 
decentralized capabilities.

The source code is available at https://github.com/cenwadike/Jarvis-Hack, and a live demo can be 
accessed at https://jarvis-hack.vercel.app/. Jarvis was developed to showcase a unique application 
of ATOM and Akash in a user-centric, voice-enabled interface.

## Key Features

- Voice command support for deploying and managing Akash cloud resources.
- Persistent storage of deployment data in PostgreSQL.
- NLP-driven interaction with the Akash Network using cosmpy and Akash CLI.
- ATOM payments for Akash provider services.

## Technical Architecture
Jarvis is a hybrid application combining a web frontend, a Python-based backend for voice and 
blockchain interaction, and external dependencies like PostgreSQL and Akash. Below is its technical 
structure:

### Frontend

- Framework: Next.js, providing a web interface for user interaction.
- Deployment: Hosted on Vercel for scalability and accessibility.

#### Components:

- Microphone input for capturing voice commands.
- Display for deployment status and resource monitoring.
- Display for payment and history.

### Backend

- Language: Python (managed via pyenv for version control).
- Voice Processing: NLP module to interpret voice commands.
- Akash Interaction:
    - cosmpy: Python library for Cosmos blockchain interactions, used to communicate with Akash 
        nodes.
    - Akash CLI: Executes commands for deployment, monitoring, and management.
- Database: PostgreSQL for storing deployment metadata (e.g., IDs, configurations, statuses).
- Wallet Integration: Interfaces with an Cosmos Hub wallet (testnet) for authentication and to send 
ATOM payments to backend.

### Blockchain Layer
- Cosmos Hub: Facilitates authentication and ATOM transactions for payments.
- Akash Network: Provides the decentralized cloud infrastructure, processing deployments funded by 
ATOM.

## Data Flow

- User speaks a command (e.g., “Deploy a web server with 2GigaByte of RAM and 500GigaByte of Storage”) 
into the microphone via the web UI.
- The frontend sends the audio to the Python backend, where NLP interprets it into an Akash action.
- The backend uses cosmpy or Akash CLI to construct and submit a deployment request, funded with 
ATOM from the user’s wallet.
- Deployment data is saved to PostgreSQL, and the frontend updates with the result.

## How it Leverages Cosmos Technologies

Jarvis integrates with the Cosmos ecosystem through ATOM and Akash, focusing on economic and 
infrastructural benefits rather than direct use of Cosmos development tools:

1. ATOM (Cosmos Hub Token)
- Use Case: Authenticate and funds Akash deployments and facilitates communication with providers.
- Implementation: Jarvis uses an Akash wallet (testnet) to send ATOM payments, processed via the 
Cosmos Hub.
- Benefit: Leverages ATOM’s utility as a payment token, simplifying transactions within the Akash 
ecosystem.

2. Akash Network (Cosmos-Based)
- Use Case: Provides the decentralized cloud platform that Jarvis manages.
- Implementation: Jarvis interfaces with Akash via cosmpy and the Akash CLI, submitting deployments 
and monitoring resources.
- Benefit: Utilizes Akash’s Cosmos-built infrastructure, benefiting from its scalability.

## Future Plans and Roadmap

Jarvis aims to evolve as a voice-driven gateway to decentralized cloud computing, with plans to enhance 
functionality and integration:

### Short-Term (Post-Hackathon, H2 2025)

- UI Polish: Improve the web interface with real-time deployment logs and voice feedback.
- Testing: Validate NLP accuracy and ATOM payment reliability on the Akash testnet.

### Medium-Term (Q1-Q3 2026)

- Voice Command Expansion: Support additional Akash operations (e.g., scaling, termination) via voice.
- Database Optimization: Add indexing and analytics for deployment history in PostgreSQL.
- Mobile Access: Develop a mobile-friendly UI or app for on-the-go management.

### Long-Term (Q4 2026 and Beyond)

- IBC Integration: Explore direct IBC use for cross-chain resource management within Cosmos.
- AI Enhancement: Train a custom NLP model for better voice command recognition in diverse accents.
- Ecosystem Collaboration: Partner with Akash and Cosmos projects to integrate Jarvis into broader 
workflows.

## Getting Started

To set up and run Jarvis locally, follow these steps:

### Prerequisites

- pyenv: Install for Python version management (curl https://pyenv.run | bash).
- PostgreSQL: Install and set up a local database (e.g., via brew install postgresql or equivalent).
- Cosmos Hub Wallet: Fund a testnet wallet with ATOM (see Akash testnet docs).
- Microphone: Ensure a working microphone for voice input.

### Setup

- Clone the Repository:

```bash
    git clone https://github.com/cenwadike/Jarvis-Hack.git
    cd Jarvis-Hack
```

### Install Backend Dependencies:

```bash
    cd backend
```

```bash
    pyenv install 3.11.9
    pyenv local 3.11.9
    
    pip install -r requirements.txt

    pip install ./en_core_web_sm-3.7.1.tar.gz

    pytest tests
```

### Configure Database:

- Initialize PostgreSQL and create a database.
- Update config with database credentials and Akash mnemonic (in a .env file).

### Run the Application:
- Start the backend: python app.py (assumed entry point; adjust as needed).

### Conclusion

Jarvis redefines decentralized cloud management by introducing voice-driven control, powered by 
ATOM payments and Akash integration. Its use of NLP, cosmpy, and the Akash CLI showcases a 
creative application within the Cosmos ecosystem. We’re excited to refine this project and invite 
the Naija HackATOM community to contribute!

Explore the project at https://github.com/cenwadike/Jarvis-Hack or try the demo at 
https://jarvis-hack.vercel.app/.
