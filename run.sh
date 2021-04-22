#!/bin/bash
# cd app
container_ip=$(hostname -i)
echo "Starting hypercorn server für container with IP ${container_ip}"
hypercorn --bind "${container_ip}:80" app
