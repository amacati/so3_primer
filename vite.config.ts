import { defineConfig } from 'vite'

export default defineConfig({
    base: '/so3_primer/',
    build: {
        outDir: 'dist',
        assetsDir: 'assets',
        sourcemap: false,
    },
})
