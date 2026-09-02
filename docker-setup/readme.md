# About Submission
The basic unit for submission is an **algorithm.**
*   Each algorithm is linked with a Docker image.
    *   Either submit the `tar.gz` file of the Docker image or
    ```plain
    docker save your_image_name:latest | gzip > your_algorithm_name.tar.gz
    ```
    *   Link with GitHub repo and use the Grand Challenge (GC) platform to build the image.
        *   Every time a commit is tagged, it will trigger the GC to build the image automatically.
        *   You can create a tag on GitHub website or using the command:
        ```plain
        git tag -a v2.1.1 -m "Test run OK locally"
        git push origin v2.1.1
        ```
*   The model can be either wrapped inside the Docker image or provided on GC by uploading the model `tar.gz` file. In this case it will be untar to `/opt/ml` in the image, where the code can access. Make sure there's no parent folder in the tar file.

```erlang
# -C means change directory first
# hence the tarfile won't have <model-dir> in it
tar -czf model.tar.gz -C <model-dir> .

# To untar and verify
tar -xzf model.tar.gz
```

*   To make things load faster in the Docker, pre-compile the your Python files to `.pyc`

```coffeescript
# Pre-compile the user directory and algorithm code
# Use pip show ... to get the user pip directory
RUN python -m compileall /home/user/.local/lib/python3.11/site-packages
RUN python -m compileall /opt/app/
```

## Docker setup for developing algorithm
![](docker-setup.png)

It can take up to an hour for GC to compile before the algorithm is usable. Hence getting error feedback from GC is time-consuming. Instead you can mock the GC behaviour locally. This will make sure the plumbing is correct (not accounting for RAM or speed though).
*   First, submit a validation run, which you'll have the input files and their structure. Download them.
*   Make a build from the Docker on the example Dockerfile (with necessary pip isntalls, etc.). We'll use it for an interactive runtime for constructing the submission code.
*   Git clone the submission algorithm repo to local, in this case, `example-photon-ct`
*   Run the Docker locally and map the input, output, model and the algorithm repo to the container. Any modification to the repo inside Docker will persist locally.

```bash
docker run --rm -it                 \
  --entrypoint bash                 \
  -v "./input:/input"               \
  -v "./output:/output"             \ 
  -v "./model:/opt/ml/model"        \
  -v "./example-photon-ct:/opt/app" \
  photon-ct
```

> Windows cmd will have problem mapping the volumes. Hence use an Ubuntu shell. The drives are under `mnt` , e.g. `/mnt/c/Users/Sun Yu/OneDrive - Peter Mac/Desktop/docker_sim`
*   Commit changes to the repo and push it back to GitHub.
*   Make tags as necessary to build the Docker on GC.
*   Wrap up the `model` folder

```erlang
tar -czf model.tar.gz -C <model-dir> .
```