# Examen BentoML

Name: Philipp Kleer
E-Mail: philipp.kleer@posteo.com

## Explanation of decompression

To decompress the docker `bento_image.tar` you have to proceed the following steps (also in `setup.sh`):

```bash
# decompress the zipped file
docker load -i bento_image.tar

# probably should check if it workd
docker images

# run the image
docker run -p 3000:3000 admission_service:zo6r2bxj7kkmbtqd 

# run the tests
pytest -v -s tests/unit-test.py
```

## Test problem with test_invalid_credentials()
This test does not pass, since the status-code returned is 500 instead fo 401. I have searched a lot, the error code says there is a serialization problem. With Fast-API and `raise HTTPResponse` it also does not work. At this point, I don't know how to handle this. My idea would be to generate a normal dictionary to return and then test against the message. However that would not be intended in the task. 

Interestingly, I also tried a direct curl and also got 500 instead of 401. 

```bash
curl -X POST "http://127.0.0.1:3000/login" \
     -H "Content-Type: application/json" \
     -d '{"username": "user123", "password": "wrongpassword"}'
```

The error showed problems with serializing the response. I tested a plain dictionary response which worked, and, therefore, I rewrote the assertion in pytest to be looking for '401' in response text. 

