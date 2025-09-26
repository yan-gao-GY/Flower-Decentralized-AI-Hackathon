# Flower Decentralized AI Hackathon

This repository provides a federated learning scenario for 2D biomedical image classification across multiple hospitals, developed for the Flower Decentralized AI Hackathon (see the [blog post](https://flower.ai/blog/2025-09-24-hackathon-sf/) and [Flower Discuss guide](https://discuss.flower.ai/t/decentralized-ai-hackathon-stanford-2025/1109)). To simulate this, we partition several [MedMNIST](https://huggingface.co/jafermarq) datasets, each representing a different hospital. These partitions are available in the `ResearchGrid` and, participants will use them to train their models.

> [!NOTE]  
> All following commands should be run from the ***project root directory***.


## Install `flwr`

Install Flower framework via:

```shell
pip install flwr
```

## Install dependencies and project (optional)

Install the dependencies specified in `pyproject.toml`, along with the `medapp` package:

```shell
pip install -e .
```

For Track 1, you do **not** need to install anything beyond `flwr`. Your code will be submitted to ResearchGrid and executed there, which means additional dependencies cannot be installed. You can find the full list of available dependencies [here](#remote-environment-dependencies).

## How to run

### Log in to your Flower account

To access the `ResearchGrid`, start by logging in with the following command:

```shell
flwr login
```

After running the command, click the login link that appears. Enter your account name and password, then confirm access for your device. Once you return to the terminal, you’ll see: `✅ Login successful.`

### Start a run

> [!NOTE]  
> For more details about CLI arguments, see the [documentation](https://flower.ai/docs/framework/ref-api-cli.html#flwr-run) or run the command with the `--help` flag.  
> You can also check out the guide [How to configure `pyproject.toml`](https://flower.ai/docs/framework/how-to-configure-pyproject-toml.html) to learn more about configuring a run.

To start a federated run with the default configuration:

```shell
# Run with the default federation and enable log streaming
flwr run --stream
```

By default, the `pathmnist` dataset (see the table in the last section) is used. After running this command, you should see logs similar to the following:

```shell
Loading project configuration... 
Success
🎊 Successfully started run 2081565958753492077
INFO :      Starting logstream for run_id `2081565958753492077`
...
```

> [!TIP]
> You can use Weights & Biases (wandb) to log your training. To do this, provide your `wandb-token`, which will be securely uploaded to the `ResearchGrid`.

To enable wandb logging:

```shell
flwr run --stream --run-config="use-wandb=true wandb-token='<your-token>'"
```

To run with a different dataset, for example `organamnist`:

```shell
# Check the table below to get `num-classes` for your target dataset.
flwr run --stream --run-config="dataset='organamnist' num-classes=11"
```

### Check the status of submitted runs

To list all submitted runs:

```shell
flwr ls
```

The output displays all runs. For example:

```shell
Loading project configuration... 
Success
📄 Listing all runs...

| Run ID              | FAB                      | Status     | Elapsed  | Created At | Running At | Finished At |
| ------------------- | ------------------------ | ---------- | -------- | ---------- | ---------- | ----------- |
| 2081565958753492077 | flwrlabs/medapp (v1.0.0) | pending    | 00:00:00 | 2025-09-26 | N/A        | N/A         |
```

### Save and pull artifacts

For the purpose of the Hackathon, it’s not necessary to save results to disk. However, if you’d like to persist results, you can write files to the directory:

```python
context.node_config["output_dir"]
```

All files placed in this directory will automatically be collected, bundled into a zip file, and uploaded to our artifacts store.

To retrieve artifacts from a run, use the following command:

```shell
flwr pull --run-id <your-run-id>
```

This will return a download link. By following the link, you can download the zip file containing all files you stored (e.g., model checkpoints, logs, or other outputs).

### Stop a run

To stop a running job:

```shell
flwr stop <your-run-id>
```

## Preview the datasets

We use [`Flower Datasets`](https://flower.ai/docs/datasets/) to create partitions of the datasets. Each dataset is defined in the `pyproject.toml` of your app. By default, `pathmnist` is used. All datasets are split into 8 partitions.

| Dataset     | # Samples | # Classes |
| ----------- | --------- | --------- |
| organamnist | 58,850    | 11        |
| pathmnist   | 107,180   | 9         |
| retinamnist | 1,600     | 5         |
| dermamnist  | 10,015    | 7         |
| bloodmnist  | 17,092    | 8         |

You can see how to save the datasets to disk and build a dataloader in the notebook: `save_datasets.ipynb`. Make sure to install the required packages first:

```shell
pip install jupyter matplotlib
```

## Appendix

### Remote environment dependencies
<details>
  <summary>See full list</summary>

  ```shell
    aiohappyeyeballs          2.6.1
    aiohttp                   3.12.15
    aiosignal                 1.4.0
    annotated-types           0.7.0
    attrs                     25.3.0
    boto3                     1.40.30
    botocore                  1.40.39
    certifi                   2025.8.3
    cffi                      2.0.0
    charset-normalizer        3.4.3
    click                     8.1.8
    contourpy                 1.3.3
    cryptography              44.0.3
    cycler                    0.12.1
    datasets                  3.1.0
    dill                      0.3.8
    evaluate                  0.4.3
    filelock                  3.13.1
    flwr                      1.23.0
    flwr-datasets             0.5.0
    fonttools                 4.60.0
    frozenlist                1.7.0
    fsspec                    2024.6.1
    gitdb                     4.0.12
    GitPython                 3.1.45
    grpcio                    1.75.0
    grpcio-health-checking    1.62.3
    hf-xet                    1.1.10
    huggingface-hub           0.35.1
    idna                      3.10
    iterators                 0.0.2
    jax                       0.5.3
    jaxlib                    0.5.3
    Jinja2                    3.1.4
    jmespath                  1.0.1
    joblib                    1.5.2
    jsonschema                4.25.1
    jsonschema-specifications 2025.9.1
    kiwisolver                1.4.9
    markdown-it-py            4.0.0
    MarkupSafe                2.1.5
    matplotlib                3.10.6
    mdurl                     0.1.2
    ml_dtypes                 0.5.3
    mpmath                    1.3.0
    msgpack                   1.1.1
    multidict                 6.6.4
    multiprocess              0.70.16
    networkx                  3.3
    numpy                     2.3.3
    opt_einsum                3.4.0
    packaging                 25.0
    pandas                    2.2.3
    pathspec                  0.12.1
    pillow                    11.0.0
    pip                       24.1.2
    platformdirs              4.4.0
    propcache                 0.3.2
    protobuf                  4.25.8
    pyarrow                   21.0.0
    pycparser                 2.23
    pycryptodome              3.23.0
    pydantic                  2.11.9
    pydantic_core             2.33.2
    Pygments                  2.19.2
    pyparsing                 3.2.5
    python-dateutil           2.9.0.post0
    pytz                      2025.2
    PyYAML                    6.0.2
    ray                       2.31.0
    referencing               0.36.2
    regex                     2025.9.18
    requests                  2.32.5
    rich                      13.9.4
    rpds-py                   0.27.1
    s3transfer                0.14.0
    safetensors               0.6.2
    scikit-learn              1.6.1
    scipy                     1.16.2
    seaborn                   0.13.2
    sentry-sdk                2.39.0
    setuptools                70.3.0
    shellingham               1.5.4
    six                       1.17.0
    smmap                     5.0.2
    sympy                     1.13.3
    threadpoolctl             3.6.0
    tokenizers                0.21.4
    tomli                     2.2.1
    tomli_w                   1.2.0
    torch                     2.8.0+cpu
    torchvision               0.23.0+cpu
    tqdm                      4.67.1
    transformers              4.51.1
    typer                     0.12.5
    typing_extensions         4.15.0
    typing-inspection         0.4.1
    tzdata                    2025.2
    urllib3                   2.5.0
    wandb                     0.21.0
    xxhash                    3.5.0
    yarl                      1.20.1
  ```
</details>
