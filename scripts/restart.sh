#!/bin/bash
echo "Restarting AI Insights Platform..."
docker-compose down
docker-compose up -d
echo "Platform restarted!"
