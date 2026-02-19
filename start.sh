#!/bin/bash
# Start JARVIS server

cd "$(dirname "$0")/backend"
source venv/bin/activate
python main.py
