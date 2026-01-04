import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],

  server: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        secure: false,
      },
    },

    // === This solves most "changes not showing" problems on Windows ===
    watch: {
      usePolling: true,        // Active polling instead of events
      interval: 800,           // Check every 0.8 seconds
      ignored: [
        '**/node_modules/**',
        '**/dist/**',
        '**/.git/**',
        '**/build/**'
      ],
    },

    hmr: {
      overlay: true,           // Show errors on screen
    },
  },

  // Faster cold start
  optimizeDeps: {
    include: ['react', 'react-dom'],
  },
})