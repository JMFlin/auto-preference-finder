import tempfile
import json

from kubernetes import client as kubernetes_client, config as kubernetes_config
from google.cloud import storage


BUCKET = "secrets.projects.bot.bucket"
BLOB = "kubeconfig"

storage_client = storage.Client()


def download_gcs_object(name_bucket, name_blob):
    bucket = storage_client.bucket(name_bucket)
    blob = bucket.blob(name_blob)
    tmp = tempfile.NamedTemporaryFile(delete=False) 
    blob.download_to_filename(tmp.name)
    return tmp.name


def load_kube_config():
    kubernetes_config.load_kube_config(config_file=download_gcs_object(BUCKET, BLOB))


def create_container_object_default():
    return kubernetes_client.V1Container(name="pi", image="perl", command=["perl", "-Mbignum=bpi", "-wle", "print bpi(1000)"]) 


def create_job_object(container):
    template = kubernetes_client.V1PodTemplateSpec(metadata=kubernetes_client.V1ObjectMeta(labels={"app": "pi"}), spec=kubernetes_client.V1PodSpec(restart_policy="Never", containers=[container]))
    return kubernetes_client.V1Job(api_version="batch/v1", kind="Job", metadata=kubernetes_client.V1ObjectMeta(name="pi"), spec=kubernetes_client.V1JobSpec(template=template, backoff_limit=4))


load_kube_config()

kubernetes_api = kubernetes_client.BatchV1Api()


def gcs_pod_trigger(request):
    job = create_job_object(create_container_object_default())
    kubernetes_api.create_namespaced_job(body=job, namespace="default")
    return json.dumps(job.to_dict())