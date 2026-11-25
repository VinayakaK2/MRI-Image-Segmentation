import type { ApiErrorPayload } from "@/types/api";

export type ApiErrorKind = "network_unreachable" | "http_error" | "invalid_response";

/** Configured FastAPI backend base URL without trailing slash. */
export const API_BASE_URL: string = (
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000"
).replace(/\/+$/, "");

const DEFAULT_TIMEOUT_MS = 10000;

/** Typed API client error carrying connectivity vs HTTP error classification. */
export class ApiClientError extends Error {
  readonly kind: ApiErrorKind;
  readonly statusCode: number | null;
  readonly errorCode: string;

  constructor(
    message: string,
    kind: ApiErrorKind,
    statusCode: number | null = null,
    errorCode = "API_ERROR"
  ) {
    super(message);
    this.name = "ApiClientError";
    this.kind = kind;
    this.statusCode = statusCode;
    this.errorCode = errorCode;
  }
}

interface FetchJsonOptions {
  signal?: AbortSignal;
  timeoutMs?: number;
}

/** Performs a typed GET request against the local FastAPI backend with abort & timeout support. */
export async function fetchApiJson<T>(
  path: string,
  options: FetchJsonOptions = {}
): Promise<T> {
  const url = `${API_BASE_URL}${path}`;
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), options.timeoutMs ?? DEFAULT_TIMEOUT_MS);

  const forwardAbort = () => controller.abort();
  if (options.signal) {
    if (options.signal.aborted) {
      window.clearTimeout(timeoutId);
      throw new DOMException("Aborted", "AbortError");
    }
    options.signal.addEventListener("abort", forwardAbort, { once: true });
  }

  try {
    const response = await fetch(url, {
      method: "GET",
      headers: { Accept: "application/json" },
      signal: controller.signal,
    });

    if (!response.ok) {
      let errCode = `HTTP_${response.status}`;
      let errMsg = `Backend returned status ${response.status}.`;
      try {
        const parsed = (await response.json()) as Partial<ApiErrorPayload>;
        if (parsed.error?.code) {
          errCode = parsed.error.code;
        }
        if (parsed.error?.message) {
          errMsg = parsed.error.message;
        }
      } catch {
        // Keep status fallback message when body is non-JSON
      }
      throw new ApiClientError(errMsg, "http_error", response.status, errCode);
    }

    try {
      return (await response.json()) as T;
    } catch {
      throw new ApiClientError(
        "Backend returned malformed JSON.",
        "invalid_response",
        response.status,
        "INVALID_JSON"
      );
    }
  } catch (err) {
    if (err instanceof ApiClientError) {
      throw err;
    }
    if (err instanceof DOMException && err.name === "AbortError") {
      // Distinguish caller cancellation from internal request timeout
      if (options.signal?.aborted) {
        throw err;
      }
      throw new ApiClientError(
        "Backend request timed out. Confirm the local FastAPI server is responsive.",
        "network_unreachable",
        null,
        "REQUEST_TIMEOUT"
      );
    }
    throw new ApiClientError(
      "Unable to connect to the local FastAPI backend at " + API_BASE_URL,
      "network_unreachable",
      null,
      "BACKEND_UNREACHABLE"
    );
  } finally {
    window.clearTimeout(timeoutId);
    if (options.signal) {
      options.signal.removeEventListener("abort", forwardAbort);
    }
  }
}
