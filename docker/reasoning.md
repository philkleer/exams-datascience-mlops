# Exam Docker

In this file, I shortly address my choices. 

For the structure of the folder, I thought it is better to keep the test files in a folder `tests` and locate the rest of the files in the main folder. As suggested, I created  a folder for the saved logs `shared_logs`. 

The structure is therefore this:

```markdown
.
├── setup.sh
├── docker-compose.yaml
├── Dockerfile
├── reasoning.md
├── tests
│   ├── authentication_test.py
│   ├── authorization_test.py
│   └── content_test.py
└── shared_logs
    └── (api_test.log) will be created
```

In the `docker-composer.yaml`, I created environment vars for the entries of user/password combination, entrypoints for the type of the machine and the results (depending on the tests).

Furthermore, I created a network so that it is easier to access the API from the other dockers. 

Finally, I linked a local volume to one in each container to append the `logs.txt` file with the results of each test. 

Instead of creating three different dockerfiles (for each test), I created a single one. In a broader more complex projet would be helpful to create single docker files (and put these also in the folder `/tests` and probably create subfolder for each test). However, this project has only three files which are not complex or large, therefore, I just copied all files into all containers. For the other approach, I must have to explicitly call the argument `dockerfile:` in the `docker-composer.yaml`.

To run it on the DataScientest machine I had to switch the python version to `slim-bullseye` since with `3.8` or `3.9` the installation of requests did not run through.