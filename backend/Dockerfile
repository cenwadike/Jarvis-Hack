# Stage 1: Build dependencies
FROM python:3.13-slim AS builder

WORKDIR /app

# Install build dependencies for Python and Akash CLI
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libc-dev \
    libpq-dev \
    python3-dev \
    curl \
    ca-certificates \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Update CA certificates to avoid SSL issues
RUN apt-get update && apt-get install -y ca-certificates && update-ca-certificates

# Upgrade pip
RUN pip install --upgrade pip --timeout 120 --retries 10

# Install Akash CLI (provider-services) with retry logic
RUN curl -sSfL https://raw.githubusercontent.com/akash-network/provider/main/install.sh | sh -s -- -b /usr/local/bin v0.6.1 || \
    (sleep 5 && curl -sSfL https://raw.githubusercontent.com/akash-network/provider/main/install.sh | sh -s -- -b /usr/local/bin v0.6.1) && \
    provider-services version

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies system-wide
RUN pip install -r requirements.txt --timeout 120 --retries 10 -v

# Copy and install the spacy model system-wide
COPY en_core_web_sm-3.7.1.tar.gz /tmp/en_core_web_sm-3.7.1.tar.gz
RUN pip install /tmp/en_core_web_sm-3.7.1.tar.gz \
    && rm /tmp/en_core_web_sm-3.7.1.tar.gz

# Stage 2: Final image
FROM python:3.13-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    postgresql \
    libpq5 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /data/pgdata

# Copy Python dependencies and Akash CLI from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Set environment variables
ENV PATH=/usr/local/bin:$PATH \
    PGDATA=/data/pgdata \
    DATABASE_URL=postgresql+psycopg2://akash_user:akash_pass@localhost/akash_manager

# Create postgres user and group
RUN groupadd -r postgres && useradd -r -g postgres -d /data/pgdata -s /bin/bash postgres

# Set ownership of the data directory
RUN chown -R postgres:postgres /data/pgdata

# Switch to postgres user for subsequent commands
USER postgres

# Initialize PostgreSQL database
RUN /usr/lib/postgresql/*/bin/initdb -D $PGDATA

# Modify PostgreSQL configuration files
RUN echo "host all all 0.0.0.0/0 md5" >> $PGDATA/pg_hba.conf && \
    echo "listen_addresses = '*'" >> $PGDATA/postgresql.conf

# Expose port
EXPOSE 5000

# Start PostgreSQL and Flask app as postgres user
CMD ["/bin/sh", "-c", "/usr/lib/postgresql/*/bin/pg_ctl -D $PGDATA start && python app.py"]
