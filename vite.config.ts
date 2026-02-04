import { defineConfig } from 'vite'

export default defineConfig({
    base: '/so3_primer_website/',
    build: {
        outDir: 'dist',
        assetsDir: 'assets',
        sourcemap: false,
    },
})
