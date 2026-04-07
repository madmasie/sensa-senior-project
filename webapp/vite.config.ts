import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Base is '/' because the site is served from a custom domain (sensa.maddiemasiello.com),
// not a GitHub Pages subpath like /sensa-senior-project/.
export default defineConfig({
  plugins: [react()],
  base: '/',
})
