import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ command }) => ({
  base: command === "build" ? "/atlas/" : "/",
  plugins: [react()],
  optimizeDeps: {
    exclude: ["maplibre-gl"],
  },
  build: {
    modulePreload: {
      resolveDependencies(filename, deps) {
        if (filename.startsWith("assets/index-")) {
          return deps.filter((dep) => !dep.includes("maplibre-"));
        }
        return deps;
      },
    },
    rolldownOptions: {
      output: {
        manualChunks(id) {
          if (id.includes("maplibre-gl") || id.includes("@maplibre")) return "maplibre";
          if (id.includes("node_modules")) return "vendor";
        },
      },
    },
  },
  server: {
    host: "127.0.0.1",
    port: 5174,
  },
}));
