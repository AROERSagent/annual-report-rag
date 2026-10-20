import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    // 把 /api 开头的请求转发到后端 8000 端口。
    // 有了代理，浏览器认为这是"同源请求"，就完全不会触发跨域拦截，
    // 比在后端配 CORS 更省事（后端那份 CORS 配置留着作为直连的兜底）。
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})
