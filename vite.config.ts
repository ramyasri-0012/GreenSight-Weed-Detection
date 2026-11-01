import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
export default defineConfig({
  plugins: [react()],
  server: {
    watch: {
      ignored: [
        "**/python_backend/.venv/**",
        "**/python_backend/runs/**",
        "**/python_backend/outputs/**",
        "**/dataset/**",
      ],
    },
  },
});
