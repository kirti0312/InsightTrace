import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development the browser talks to this dev server, and /api/... is passed on to the Python API,
// so no CORS settings are needed locally. For a deployed site set VITE_API_URL (see .env.example).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": { target: process.env.VITE_PROXY_TARGET || "http://127.0.0.1:8000", changeOrigin: true } },
  },
});
