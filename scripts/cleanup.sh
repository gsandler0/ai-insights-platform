#!/bin/bash
echo "Cleaning up AI Insights Platform..."
docker-compose down -v
docker-compose rm -f
echo "Cleanup completed."
