FROM node:22-alpine AS build

WORKDIR /app

# npm ci against the committed lockfile: the previous `npm install` from a
# bare package.json could resolve different transitive versions on every
# image build, so the built bundle wasn't reproducible.
COPY package.json package-lock.json ./
RUN --mount=type=cache,target=/root/.npm \
    npm ci

COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
