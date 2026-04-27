# @sundrift/config

Shared configuration presets for the Sundrift monorepo.

## Exports

- `@sundrift/config` — runtime helpers (currently `parseEnv`, `EnvValidationError`)
- `@sundrift/config/eslint` — ESLint flat config preset
- `@sundrift/config/prettier` — Prettier preset
- `@sundrift/config/tsconfig.base.json` — base `tsconfig` for extending

## Usage

### TypeScript

```jsonc
// apps/<app>/tsconfig.json
{
  "extends": "@sundrift/config/tsconfig.base.json",
  "compilerOptions": {
    "rootDir": "./src",
    "outDir": "./dist",
  },
  "include": ["src/**/*"],
}
```

### ESLint

```js
// apps/<app>/eslint.config.js
import config from "@sundrift/config/eslint";
export default config;
```

### Prettier

```js
// apps/<app>/prettier.config.js
export { default } from "@sundrift/config/prettier";
```

### Env validation

```ts
import { parseEnv } from "@sundrift/config";
import { z } from "zod";

export const env = parseEnv(
  z.object({
    DATABASE_URL: z.string().url(),
    GROWATT_API_TOKEN: z.string().min(1),
  }),
);
```

Always import this in a single, eagerly-evaluated module so misconfiguration
crashes at startup, not deep inside a request handler.
