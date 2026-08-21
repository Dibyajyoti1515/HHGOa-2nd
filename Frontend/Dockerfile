# ============================================================
# Frontend image — React + Vite dev server.
#
# ASSUMPTION (unverified: GitHub wouldn't let me fetch
# Frontend/package.json): standard Vite scaffold with a "dev" script
# that runs `vite`. If your package.json's dev script differs, only
# the CMD line at the bottom needs to change.
# ============================================================
FROM node:20-alpine

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

EXPOSE 5173

# --host 0.0.0.0 is required so Vite's dev server is reachable from
# outside the container (it binds to localhost only by default).
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "5173"]