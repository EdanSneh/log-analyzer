# Running the Argo workflow

the argo workflow can be run by going to the ip at:

```sh
kubectl get svc argo-artifacts-minio  --template="{{range .status.loadBalancer.ingress}}{{.ip}}{{end}}"
#port 9000
#user AKIAIOSFODNN7EXAMPLE
#pass wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
```

This will get you to the minio server. Go to the log-analyzer bucket at /input/ and enter a txt file.

When putting in a txt file an argo event is triggered which will deploy a workflow which retriggers the argo event.

The argo events file can be found at [yamls/webhook_logs.yaml](yamls/webhook_logs.yaml). If the webhook has been killed accidently, redeploy it with [yamls/nowebhook_logs.yaml](yamls/nowebhook_logs.yaml)

##### Notes: The argo event is currently the weakest link of this program since it does not get set up immediately so putting multiple files in minio will not work.

The argo workflow will also deploy kubeflow's tfserver, and tfsvc which will allow the model to be pinged by the tfclient. The tfserver takes in an input from the bucket at gs://argo-ml-demo (do not delete folder 0). It will take in the folder with the largest number signifying the newest version number. Each of these folders contains an output of [serve.py](serve.py). To learn more about how serve.py works, check out [TFserving](https://github.com/argoproj/argo-ml-demo/blob/master/docs/TFSERVE.md).

TFserving is only nessesary for deploying the finished trained model, and analyze.py can be used in the meantime.

The TF client handles all of the non-tensorflow output for TFserving, making it useful and readable. It can continuously call the TFserver.

Lastly the TFJob will train the model with the given logs after it checks out the client.

## Things that must still get done

1. Model needs to be optimized for distinguishing common logs
2. Once model is finished, it should be TF-served for production
2. Workflow needs conditional step to send emails when uncommon log detected and to decide whether to trigger TFJob
3. [train.py](train.py) must automatically update the gs bucket (without leaking credentials to github)
4. **Optional:** For release, minio event should be changed to s3/gs