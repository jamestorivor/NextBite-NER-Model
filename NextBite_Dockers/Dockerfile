FROM python:3.10-slim

# Set working directory
WORKDIR /code

# Install dependencies
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Create non-root user
RUN useradd -m user

# Set environment variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Switch to the new user
USER user

# Set the working directory for the user
WORKDIR $HOME/app

# Copy app code to container
COPY --chown=user . $HOME/app

# Command to run the app
CMD uvicorn app:app --host 0.0.0.0 --port 7860
