import { type NextRequest } from "next/server"

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { page_id, conversation_id, tools, llm } = body

    // Get the authorization header from the request
    const authHeader = request.headers.get("authorization")

    if (!authHeader) {
      return new Response(JSON.stringify({ error: "Authorization header is required" }), { 
        status: 401,
        headers: { "Content-Type": "application/json" }
      })
    }

    // Forward the request to the backend - it returns a streaming response
    const response = await fetch(`http://gira-agent:8081/api/v1/query/regenerate_response`, {
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
      return new Response(
        JSON.stringify({
          error: errorData.message || errorData.detail || "Regenerate request failed",
          status: response.status,
        }),
        { 
          status: response.status,
          headers: { "Content-Type": "application/json" }
        }
      )
    }

    // Pass through the streaming response from backend
    return new Response(response.body, {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
      },
    })
  } catch (error) {
    console.error("Regenerate API error:", error)
    return new Response(JSON.stringify({ error: "Internal server error" }), { 
      status: 500,
      headers: { "Content-Type": "application/json" }
    })
  }
}
