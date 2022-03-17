docker build -t pipeline-mnist-train-faiss:1.0.0 .

docker tag pipeline-mnist-train-faiss:1.0.0 kubeflow-registry.default.svc.cluster.local:30000/pipeline-mnist-train-faiss:1.0.0

docker push kubeflow-registry.default.svc.cluster.local:30000/pipeline-mnist-train-faiss:1.0.0
