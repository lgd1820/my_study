docker build -t pipeline-mnist-embedding:1.0.0 .

docker tag pipeline-mnist-embedding:1.0.0 kubeflow-registry.default.svc.cluster.local:30000/pipeline-mnist-embedding:1.0.0

docker push kubeflow-registry.default.svc.cluster.local:30000/pipeline-mnist-embedding:1.0.0
