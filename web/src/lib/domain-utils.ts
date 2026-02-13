/**
 * Domain-based Brand Detection Utility
 *
 * Extracts brand ID from subdomain:
 * - richesseclub.ontix.co.kr -> richesseclub
 * - admin.ontix.co.kr -> null (super admin)
 * - api.ontix.co.kr -> null (API)
 */

// Reserved subdomains that are not brand IDs
const RESERVED_SUBDOMAINS = ["admin", "api", "www", "localhost"];

/**
 * Extract brand ID from hostname
 */
export function getBrandIdFromDomain(hostname?: string): string | null {
  if (!hostname) {
    if (typeof window !== "undefined") {
      hostname = window.location.hostname;
    } else {
      return null;
    }
  }

  // Local development
  if (hostname === "localhost" || hostname === "127.0.0.1") {
    return null;
  }

  // Check if it's a subdomain of ontix.co.kr
  const match = hostname.match(/^([^.]+)\.ontix\.co\.kr$/);
  if (match) {
    const subdomain = match[1].toLowerCase();

    // Skip reserved subdomains
    if (RESERVED_SUBDOMAINS.includes(subdomain)) {
      return null;
    }

    return subdomain;
  }

  // Direct IP access or other domains
  return null;
}

/**
 * Check if current domain is admin dashboard
 */
export function isAdminDomain(hostname?: string): boolean {
  if (!hostname) {
    if (typeof window !== "undefined") {
      hostname = window.location.hostname;
    } else {
      return false;
    }
  }

  return hostname === "admin.ontix.co.kr";
}

/**
 * Check if running in local development
 */
export function isLocalDevelopment(): boolean {
  if (typeof window === "undefined") return false;
  const hostname = window.location.hostname;
  return hostname === "localhost" || hostname === "127.0.0.1";
}

/**
 * Get the appropriate redirect URL for a brand
 */
export function getBrandDashboardUrl(brandId: string): string {
  if (isLocalDevelopment()) {
    return `/brand-dashboard`;
  }
  return `https://${brandId}.ontix.co.kr/brand-dashboard`;
}
