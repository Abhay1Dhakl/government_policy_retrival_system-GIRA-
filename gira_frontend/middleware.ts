import { NextResponse } from "next/server"
import type { NextRequest } from "next/server"

export function middleware(request: NextRequest) {
  // Only run middleware for protected routes
  if (request.nextUrl.pathname.startsWith("/chat")) {
    const token = request.cookies.get("authToken")?.value

    if (!token) {
      // No token, redirect to login
      return NextResponse.redirect(new URL("/login", request.url))
    }

    // Token exists, proceed to chat page
  }

  if (request.nextUrl.pathname.startsWith("/users")) {
    const token = request.cookies.get("authToken")?.value

    if (!token) {
      // No token, redirect to login
      return NextResponse.redirect(new URL("/login", request.url))
    }
  }

  return NextResponse.next()
}

export const config = {
  matcher: ["/chat/:path*", "/users/:path*"],
}
