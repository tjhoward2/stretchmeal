import type { z } from "zod";

export class EnvValidationError extends Error {
  constructor(
    message: string,
    public readonly issues: z.ZodIssue[],
  ) {
    super(message);
    this.name = "EnvValidationError";
  }
}

/**
 * Parse and validate environment variables against a Zod schema.
 * Throws EnvValidationError with a human-readable message on failure.
 */
export function parseEnv<S extends z.ZodTypeAny>(
  schema: S,
  source: Record<string, string | undefined> = process.env,
): z.infer<S> {
  const result = schema.safeParse(source);
  if (!result.success) {
    const lines = result.error.issues.map((issue) => {
      const path = issue.path.length > 0 ? issue.path.join(".") : "(root)";
      return `  - ${path}: ${issue.message}`;
    });
    throw new EnvValidationError(
      `Invalid environment variables:\n${lines.join("\n")}`,
      result.error.issues,
    );
  }
  return result.data;
}
