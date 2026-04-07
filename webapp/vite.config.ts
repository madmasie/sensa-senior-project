import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Base path must match the GitHub repository name so asset URLs resolve
// correctly when served from https://<user>.github.io/<repo>/
export default defineConfig({
  plugins: [react()],
  base: '/sensa-senior-project/',
})
