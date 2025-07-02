FROM python:3.11-slim

WORKDIR /app

# Install dependencies and configure locale for UTF-8 support
RUN apt-get update && apt-get install -y \
    curl \
    locales \
    && rm -rf /var/lib/apt/lists/* \
    && sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen \
    && locale-gen

# Set locale environment variables
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8
ENV PYTHONIOENCODING=utf-8

COPY requirements-frontend.txt .
RUN pip install --no-cache-dir -r requirements-frontend.txt

COPY streamlit_app.py .

RUN mkdir -p ~/.streamlit /app/logs /app/config

RUN echo "\
[server]\n\
headless = true\n\
enableCORS = false\n\
port = 8501\n\
[theme]\n\
base = \"dark\"\n\
" > ~/.streamlit/config.toml

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "streamlit_app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
