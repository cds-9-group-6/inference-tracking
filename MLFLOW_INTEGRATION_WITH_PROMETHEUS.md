# setup metrics tracking in prometheus and grafana dashboard 

## Command to start the prometheus locally

```bash
podman run --platform linux/arm64 \
  --name my-prometheus \
  --mount type=bind,source=$(pwd)/prometheus.yml,destination=/etc/prometheus/prometheus.yml \
  -v prometheus-data:/prometheus \
  -p 9090:9090 \
  -d \
  prom/prometheus:latest \
  --config.file=/etc/prometheus/prometheus.yml \
  --web.enable-otlp-receiver \
  --web.enable-remote-write-receiver \
  --enable-feature=otlp-deltatocumulative
```

podman run --platform linux/arm64 \
  --name my-prometheus \
  --mount type=bind,source=$(pwd)/prometheus.yml,destination=/etc/prometheus/prometheus.yml \
  -v prometheus-data:/prometheus \
  -p 9090:9090 \
  -d \
  prom/prometheus:latest \
  --config.file=/etc/prometheus/prometheus.yml \
  --web.enable-otlp-receiver \
  --enable-feature=otlp-deltatocumulative


podman run --platform linux/arm64 \
  -d --name=grafana \
  -p 3000:3000 \
  --mount type=bind,source=$(pwd)/grafana/grafana.ini,destination=/etc/grafana/grafana.ini \
  --mount type=bind,source=$(pwd)/grafana/grafana_provisioning,destination=/etc/grafana/provisioning \
  -v grafana-data:/var/lib/grafana \
  grafana/grafana-oss:latest





podman run -d \
  --name otel-collector \
  --rm \
  --platform linux/arm64 \
  --mount type=bind,source=$(pwd)/otel_collector_config.yml,destination=/etc/otelcol-contrib/config.yaml \
  -p 4317:4317 \
  -p 4318:4318 \
  -p 8889:8889 \
  otel/opentelemetry-collector-contrib:latest