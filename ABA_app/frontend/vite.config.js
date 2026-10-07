import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Build output goes to frontend/dist/, which FastAPI serves (see app/config.py::DIST_DIR).
// In dev, `npm run dev` proxies /api to the uvicorn server so both run side by side.
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "dist",
    emptyOutDir: true,
    // Cytoscape is most of the app's code and only the graph views need it: its own chunk,
    // fetched when Graph Studio or a Review graph is opened.
    rollupOptions: {
      output: { manualChunks: { cytoscape: ["cytoscape", "cytoscape-dagre", "cytoscape-fcose"] } },
    },
    chunkSizeWarningLimit: 700,
  },
  server: {
    port: 5173,
    proxy: { "/api": { target: "http://127.0.0.1:8000", changeOrigin: true } },
  },
});
