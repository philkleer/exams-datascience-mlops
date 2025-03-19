#!/bin/bash

# decompress the zipped file
docker load -i bento_image.tar

# probably should check if it workd
docker images

# run the image
docker run -p 3000:3000 admission_service:2z3fndhgk6vsjtqd

# run the tests
pytest -v -s tests/unit-test.py > logs/logs.txt 2>&1
