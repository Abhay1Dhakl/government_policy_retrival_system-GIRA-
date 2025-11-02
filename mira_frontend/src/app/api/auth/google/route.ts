import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { credential } = body;

    if (!credential) {
      return NextResponse.json(
        { error: 'Google credential is required' },
        { status: 400 }
      );
    }

    const apiBaseUrl = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8082/api/v1';

    console.log('Sending token to backend:', `${apiBaseUrl}/token/oauth/callback/`);

    // Send the Google OAuth token to backend
    const response = await fetch(`${apiBaseUrl}/token/oauth/callback/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        token: credential,
      }),
    });

    const data = await response.json();
    console.log('Backend response:', data);

    if (!response.ok) {
      return NextResponse.json(
        { 
          error: data.message || data.detail || 'Google sign-in failed',
          details: data
        },
        { status: response.status }
      );
    }

    return NextResponse.json(data, { status: response.status });

  } catch (error) {
    console.error('Google OAuth API error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
};