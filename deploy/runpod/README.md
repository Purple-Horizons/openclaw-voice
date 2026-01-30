# Deploy OpenClaw Voice on RunPod

Deploy OpenClaw Voice on RunPod for GPU-accelerated voice inference.

## Quick Deploy

### 1. Create a RunPod Account
Sign up at [runpod.io](https://runpod.io)

### 2. Deploy from Template

Use our pre-built template:

```
Template ID: openclaw-voice
Docker Image: ghcr.io/purple-horizons/openclaw-voice:latest
```

Or deploy manually:

### 3. Manual Deployment

**Create a new Pod:**
- GPU: RTX 4090 recommended ($0.44/hr)
- Docker Image: `ghcr.io/purple-horizons/openclaw-voice:latest`
- Expose Port: 8765
- Volume: 20GB (for model cache)

**Environment Variables:**
```
OPENCLAW_STT_MODEL=large-v3-turbo
OPENCLAW_STT_DEVICE=cuda
OPENCLAW_REQUIRE_AUTH=true
OPENCLAW_MASTER_KEY=<generate-a-secure-key>
OPENAI_API_KEY=<your-openai-key>
```

### 4. Connect

Once deployed, get your Pod's public URL:
```
wss://<pod-id>-8765.proxy.runpod.net/ws?api_key=<your-key>
```

## Cost Optimization

### On-Demand (Pay per hour)
- RTX 4090: ~$0.44/hr
- RTX 3090: ~$0.30/hr
- A10: ~$0.50/hr

### Spot Instances (Up to 80% cheaper)
- Use spot instances for non-critical workloads
- May be interrupted with 30s notice

### Reserved (Best for production)
- Reserve a GPU for consistent pricing
- No interruptions

## Scaling

### Single Pod
- Handles ~10-20 concurrent voice sessions
- Good for testing and small deployments

### Multiple Pods with Load Balancer
1. Deploy 2+ pods
2. Use RunPod's serverless endpoints
3. Or put behind Cloudflare Load Balancer

## Monitoring

### Health Check
```bash
curl https://<pod-url>/
```

### Check Logs
```bash
runpodctl logs <pod-id>
```

## Troubleshooting

### GPU Not Detected
- Ensure NVIDIA drivers are loaded
- Check `nvidia-smi` in pod terminal

### Out of Memory
- Reduce batch size
- Use smaller Whisper model (base vs large)
- Upgrade to GPU with more VRAM

### High Latency
- Use larger GPU
- Enable streaming responses
- Check network latency to RunPod region
