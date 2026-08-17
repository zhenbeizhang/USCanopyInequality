# 交互地图部署模板

本目录提供可复用的部署示例，不保存真实服务器地址、账号、私钥路径或域名。部署前请在服务器和本地终端中自行设置这些值，不要将凭据提交到 Git。

## 推荐入口

正式部署建议提供三个 HTTPS 入口：

1. `https://atlas.example.org/`：交互地图；
2. `https://atlas.example.org/api/health`：API 健康检查；
3. `https://atlas.example.org/api/...`：受 Nginx 代理保护的瓦片接口。

## 服务器目录

- 地图前端：`/srv/who-has-shade/atlas`
- FastAPI：`/srv/who-has-shade/api`
- 城市 COG：`/srv/who-has-shade/data/cogs`
- Nginx 配置：`/etc/nginx/sites-available/who-has-shade`
- API 服务：`who-has-shade-api.service`

## 上传 COG

使用非 root 部署账号，并显式传入服务器、私钥和本地 COG 目录：

```powershell
.\deploy\upload_cogs.ps1 `
  -Server "atlas.example.org" `
  -RemoteUser "deploy" `
  -KeyPath "$HOME\.ssh\atlas_deploy_key" `
  -CogRoot "D:\path\to\cogs"
```

脚本会按远程文件大小跳过已上传城市，网络中断后可重复运行。上传完成后，在服务器上运行 `deploy/verify_cogs.py` 校验城市清单与 COG 结构。

## 上线检查

```bash
systemctl status nginx who-has-shade-api
journalctl -u who-has-shade-api -n 100 --no-pager
nginx -t
curl -fsS http://127.0.0.1:8000/api/health
```

将示例域名替换为实际域名，在 systemd 服务中设置同源的 `ATLAS_ALLOWED_ORIGINS`，再用 Certbot 或云平台证书启用 HTTPS。
