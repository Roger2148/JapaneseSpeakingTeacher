import fs from "node:fs";
import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");

  const keyPath = env.VITE_DEV_HTTPS_KEY?.trim();
  const certPath = env.VITE_DEV_HTTPS_CERT?.trim();
  const forceHttps = env.VITE_DEV_HTTPS?.trim().toLowerCase() === "true";

  let https: true | { key: Buffer; cert: Buffer } | undefined;
  if (
    keyPath &&
    certPath &&
    fs.existsSync(keyPath) &&
    fs.existsSync(certPath)
  ) {
    https = {
      key: fs.readFileSync(keyPath),
      cert: fs.readFileSync(certPath)
    };
  } else if (forceHttps) {
    https = true;
  }

  return {
    plugins: [react()],
    server: https
      ? {
          https
        }
      : undefined
  };
});
