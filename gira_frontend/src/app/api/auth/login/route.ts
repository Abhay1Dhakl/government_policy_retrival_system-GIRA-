import { NextRequest, NextResponse } from 'next/server';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://gira-backend:8082/api/v1/';

// Function to get CSRF token from Django
async function getCSRFToken(): Promise<string | null> {
  try {
    const response = await fetch(`${API_BASE_URL.replace('/api/v1', '')}/`, {
      method: 'GET',
      credentials: 'include',
    });

    const cookies = response.headers.get('set-cookie');
    if (cookies) {
      const csrfMatch = cookies.match(/csrftoken=([^;]+)/);
      if (csrfMatch) {
        return csrfMatch[1];
      }
    }

    // Alternative: try to get from a dedicated CSRF endpoint if available
    const csrfResponse = await fetch(`${API_BASE_URL.replace('/api/v1', '')}/csrf/`, {
      method: 'GET',
    }).catch(() => null);

    if (csrfResponse?.ok) {
      const csrfData = await csrfResponse.json();
      return csrfData.csrfToken || csrfData.token || null;
    }

    return null;
  } catch (error) {
    console.error('Failed to get CSRF token:', error);
    return null;
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    console.log('Proxying login request to:', `${API_BASE_URL}/token/`);
    console.log('Request body:', { email: body.email, password: '[HIDDEN]' });

    // Try to get CSRF token
    const csrfToken = await getCSRFToken();
    console.log('CSRF token obtained:', csrfToken ? 'Yes' : 'No');

    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
      'User-Agent': 'NextJS-Proxy/1.0',
    };

    // Add CSRF token if available
    if (csrfToken) {
      headers['X-CSRFTOKEN'] = csrfToken;
      headers['X-CSRFToken'] = csrfToken; // Some APIs expect this format
    }

    const response = await fetch(`${API_BASE_URL}/token/`, {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
    });

    console.log('Backend response status:', response.status);
    console.log('Backend response headers:', Object.fromEntries(response.headers.entries()));

    const data = await response.json();
    console.log('Backend response data:', data);

    if (!response.ok) {
      console.error('Backend API error:', {
        status: response.status,
        statusText: response.statusText,
        data: data
      });

      return NextResponse.json(
        {
          error: data.message || data.detail || data.error || 'Authentication failed',
          status: response.status,
          details: data
        },
        { status: response.status }
      );
    }

    console.log('Login successful, returning data');
    return NextResponse.json(data);
  } catch (error) {
    console.error('Auth proxy error:', error);
    return NextResponse.json(
      {
        error: 'Internal server error',
        details: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 500 }
    );
  }
}

// Handle OPTIONS request for CORS preflight
export async function OPTIONS() {
  return new NextResponse(null, {
    status: 200,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, X-CSRFTOKEN',
    },
  });
}