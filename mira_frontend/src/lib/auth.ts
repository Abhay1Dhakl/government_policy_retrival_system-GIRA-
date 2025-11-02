interface LoginResponse {
  status: number
  message: string
  data: {
    access_token: string
    refresh_token: string
    has_details: boolean
  }
}

interface LoginCredentials {
  email: string
  password: string
}

interface CreatePasswordData {
  email: string
  password: string
}

interface CreatePasswordResponse {
  status: number
  message: string
  data?: any
}

interface GoogleSignInResponse {
  status: number
  message: string
  data: {
    access_token: string
    refresh_token: string
    has_details: boolean
  }
}

interface JWTPayload {
  profile_completed?: boolean
  [key: string]: any
}

interface RegenerateResponseData {
  page_id: string
  conversation_id: string
  tools: string[]
  llm: string
}

interface FeedbackData {
  conversation_id: string
  user_query: string
  assistant_response: string
  feedback: number // 1 for like, -1 for dislike
  feedback_reason: string
}

class AuthService {
  private baseUrl: string

  constructor() {
    this.baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8082/api/v1"
  }

  async login(credentials: LoginCredentials): Promise<LoginResponse> {
    const response = await fetch("/api/auth/login", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(credentials),
    })

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}))
      throw new Error(
        errorData.error || errorData.message || errorData.detail || `Login failed: ${response.statusText}`,
      )
    }

    const data = await response.json()

    if (data.status !== 200 || !data.data || !data.data.access_token) {
      throw new Error(data.message || "Invalid response from server")
    }

    this.setHasDetails(data.data.has_details)
    return data
  }

  async createPassword(data: CreatePasswordData): Promise<CreatePasswordResponse> {
    const response = await fetch("/api/auth/create-password", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(data),
    })

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}))
      throw new Error(
        errorData.error || errorData.message || errorData.detail || `Password creation failed: ${response.statusText}`,
      )
    }

    const responseData = await response.json()
    return responseData
  }

  async googleSignIn(credential: string): Promise<GoogleSignInResponse> {
    const response = await fetch("/api/auth/google", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ credential }),
    })

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}))
      throw new Error(
        errorData.error || errorData.message || errorData.detail || `Google sign-in failed: ${response.statusText}`,
      )
    }

    const data = await response.json()

    if (data.status !== 200 || !data.data || !data.data.access_token) {
      throw new Error(data.message || "Invalid response from server")
    }

    // Store tokens and has_details just like in regular login
    this.setToken(data.data.access_token)
    if (data.data.refresh_token) {
      this.setRefreshToken(data.data.refresh_token)
    }
    this.setHasDetails(data.data.has_details ?? false)

    return data
  }

  setToken(token: string): void {
    if (typeof window !== "undefined") {
      localStorage.setItem("auth_token", token)
    }
  }

  getToken(): string | null {
    if (typeof window !== "undefined") {
      return localStorage.getItem("auth_token")
    }
    return null
  }

  removeToken(): void {
    if (typeof window !== "undefined") {
      localStorage.removeItem("auth_token")
      localStorage.removeItem("refresh_token")
      localStorage.removeItem("has_details")
    }
  }

  setRefreshToken(refreshToken: string): void {
    if (typeof window !== "undefined") {
      localStorage.setItem("refresh_token", refreshToken)
    }
  }

  getRefreshToken(): string | null {
    if (typeof window !== "undefined") {
      return localStorage.getItem("refresh_token")
    }
    return null
  }

  isAuthenticated(): boolean {
    return this.getToken() !== null
  }

  // Logout user
  logout(): void {
    this.removeToken()
  }

  // Get authorization header for API requests
  getAuthHeader(): Record<string, string> {
    const token = this.getToken()
    if (token) {
      return {
        Authorization: `Bearer ${token}`,
      }
    }
    return {}
  }

  private decodeToken(token: string): JWTPayload | null {
    try {
      const base64Url = token.split(".")[1]
      const base64 = base64Url.replace(/-/g, "+").replace(/_/g, "/")
      const jsonPayload = decodeURIComponent(
        atob(base64)
          .split("")
          .map((c) => "%" + ("00" + c.charCodeAt(0).toString(16)).slice(-2))
          .join(""),
      )
      return JSON.parse(jsonPayload)
    } catch (error) {
      console.error("Error decoding token:", error)
      return null
    }
  }

  isProfileCompleted(): boolean {
    if (typeof window !== "undefined") {
      const hasDetails = localStorage.getItem("has_details")
      return hasDetails === "true"
    }
    return false
  }

  // Store has_details status from login response
  setHasDetails(hasDetails: boolean): void {
    if (typeof window !== "undefined") {
      localStorage.setItem("has_details", hasDetails.toString())
    }
  }

  async regenerateResponse(data: RegenerateResponseData): Promise<any> {
    const response = await fetch("/api/regenerate-response", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${this.getToken()}`,
      },
      body: JSON.stringify(data),
    })

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}))
      throw new Error(
        errorData.error || errorData.message || errorData.detail || `Regenerate failed: ${response.statusText}`,
      )
    }

    return await response.json()
  }

  async storeFeedback(data: FeedbackData): Promise<any> {
    const response = await fetch("/api/store-feedback", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${this.getToken()}`,
      },
      body: JSON.stringify(data),
    })

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}))
      throw new Error(
        errorData.error || errorData.message || errorData.detail || `Feedback failed: ${response.statusText}`,
      )
    }

    return await response.json()
  }
}

export const authService = new AuthService()
export default authService