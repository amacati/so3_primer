import { defineConfig } from 'vite'

export default defineConfig({
    base: '/so3_primer/',
    build: {
        outDir: 'dist',
        assetsDir: 'assets',
        sourcemap: false,
        chunkSizeWarningLimit: 600,
        rollupOptions: {
            output: {
                manualChunks(id) {
                    if (id.includes('node_modules/three')) {
                        return 'three';
                    }
                    if (id.includes('node_modules/d3')) {
                        return 'd3';
                    }
                }
            }
        }
    },
})
