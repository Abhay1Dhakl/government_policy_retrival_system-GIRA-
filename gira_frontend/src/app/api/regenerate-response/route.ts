import { type NextRequest, NextResponse } from "next/server"

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { page_id, conversation_id, tools, llm } = body

    // Get the authorization header from the request
    const authHeader = request.headers.get("authorization")

    if (!authHeader) {
      return NextResponse.json({ error: "Authorization header is required" }, { status: 401 })
    }

    const response = await fetch(`${process.env.NEXT_PUBLIC_CHAT_API_BASE_URL}/query/regenerate_response`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: authHeader,
      },
      body: JSON.stringify({
        page_id,
        conversation_id,
        tools,
        llm,
      }),
    })

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}))
      return NextResponse.json(
        {
          error: errorData.message || errorData.detail || "Regenerate request failed",
          status: response.status,
        },
        { status: response.status },
      )
    }

    const data = await response.json()
    return NextResponse.json(data)
  } catch (error) {
    console.error("Regenerate API error:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}
