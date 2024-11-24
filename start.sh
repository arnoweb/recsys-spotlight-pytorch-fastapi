#!/bin/bash
sleep 10 && solara run app.py --host=0.0.0.0 --port=80 &
(nohup uvicorn application.api.app:app --host=0.0.0.0 --port=8765)