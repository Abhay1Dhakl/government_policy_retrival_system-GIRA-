import { type NextRequest, NextResponse } from "next/server"

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const authHeader = request.headers.get("authorization")

    const baseCandidates = [
      process.env.CHAT_API_BASE_URL,
      process.env.NEXT_PUBLIC_CHAT_API_BASE_URL,
      "http://gira-agent:8081/api/v1",
      "http://localhost:8081/api/v1",
    ].filter((v): v is string => !!v)

    let lastError: unknown = null

    for (const candidate of baseCandidates) {
      const baseUrl = candidate.replace(/\/$/, "")
      const url = `${baseUrl}/feedback/store_feedback`

      try {
        const response = await fetch(url, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            ...(authHeader && { Authorization: authHeader }),
          },
          body: JSON.stringify(body),
        })

        if (!response.ok) {
          const errorText = await response.text()
          console.error(`Backend error via ${url}:`, errorText)
          return NextResponse.json({ error: errorText || "Failed to store feedback" }, { status: response.status })
        }

        const data = await response.json()
        return NextResponse.json(data)
      } catch (err) {
        lastError = err
        console.error(`Fetch failed via ${url}:`, err)
        // Try next candidate
      }
    }

    return NextResponse.json({ error: (lastError as Error)?.message || "Failed to reach feedback service" }, { status: 502 })
  } catch (error) {
    console.error("Store feedback error:", error)
    return NextResponse.json({ error: (error as Error).message || "Internal server error" }, { status: 500 })
  }
}
